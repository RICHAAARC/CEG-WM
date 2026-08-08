"""CPU contracts for the real HF transport diagnostic execution path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

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
    canonical_digest,
    load_hf_transmission_protocol,
)
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.runners.development_persistence import CommittedUnit
from experiments.runners.hf_transmission_diagnostic import (
    HfTransmissionDiagnosticRunner,
    HfTransmissionRunnerError,
)
from main import identify_root_key
from runtime import create_runtime_adapter
from runtime import RuntimeDeviceCapabilities, Sd35RuntimeAdapter
from scripts.experiment_execution import hf_transmission_diagnostic_entrypoint
from tests.unit.test_runtime_content_write_and_vae import (
    FakeContentBackend,
    _base_latent as small_base_latent,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/hf_transmission_diagnostic.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-hf-transmission-diagnostic-test-key"


class _CudaDiagnosticBackend(FakeContentBackend):
    def __init__(
        self,
        *,
        failure_run_index: int | None = None,
        failure_factory: Callable[[], Exception] | None = None,
    ) -> None:
        callbacks = tuple(tuple(range(20)) for _ in range(24))
        super().__init__(callback_sequences=callbacks)  # type: ignore[arg-type]
        self.failure_run_index = failure_run_index
        self.failure_factory = failure_factory

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(cpu_available=True, cuda_device_count=1)

    def set_development_generation_prompts(self, _prompt: str) -> None:
        return None

    def run_generation(self, initial_latent, callback):
        if self.run_calls == self.failure_run_index:
            self.run_calls += 1
            assert self.failure_factory is not None
            raise self.failure_factory()
        return super().run_generation(initial_latent, callback)


def _install_entrypoint_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backends: list[_CudaDiagnosticBackend],
) -> None:
    package = tmp_path / "hf_transmission_test_package.zip"
    package.write_bytes(b"checked-in-test-package")
    monkeypatch.setattr(
        hf_transmission_diagnostic_entrypoint,
        "Sd35PipelineBackend",
        lambda **_kwargs: backends.pop(0),
    )
    initialize_runtime = Sd35RuntimeAdapter.initialize
    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "initialize",
        lambda self, _requested_device: initialize_runtime(self, "cpu"),
    )
    monkeypatch.setattr(
        hf_transmission_diagnostic_entrypoint,
        "_build_or_verify_package",
        lambda *_args: package,
    )
    monkeypatch.setattr(
        hf_transmission_diagnostic_entrypoint,
        "_base_latent",
        lambda *_args, **_kwargs: small_base_latent(),
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _index: 1)


def _load_terminal_scientific_evidence(
    run_root: Path,
) -> tuple[tuple[DevelopmentScientificRecord, CommittedUnit], ...]:
    markers = tuple(
        CommittedUnit(**json.loads(path.read_text("utf-8")))
        for path in (run_root / "markers").glob("*.COMMITTED.json")
    )
    latest_by_unit: dict[str, CommittedUnit] = {}
    for marker in sorted(markers, key=lambda item: item.attempt_index):
        if marker.record_kind == "development_scientific_record":
            latest_by_unit[marker.unit_id] = marker
    evidence = []
    for marker in sorted(latest_by_unit.values(), key=lambda item: item.unit_index):
        with ZipFile(
            run_root / "bundles" / f"sha256_{marker.bundle_sha256}.zip"
        ) as source:
            record = DevelopmentScientificRecord.from_payload(
                json.loads(
                    source.read("records/development_scientific_record.json")
                )
            )
        evidence.append((record, marker))
    return tuple(evidence)


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
    assert observations[-1]["registered_template_digest"] != observations[-1][
        "wrong_key_template_digest"
    ]
    assert observations[-1]["registered_key_role"] == "registered"
    assert observations[-1]["wrong_key_key_role"] == "wrong"
    assert observations[-1]["wrong_key_index"] == 0
    formal_results = record.operation_result_payload["formal_hf_detection_results"]
    assert formal_results["registered"]["candidate_id"] == "hf_sparse_tail"
    assert (
        formal_results["registered"]["detector_config_digest"]
        == formal_results["wrong_key"]["detector_config_digest"]
        == formal_results["primary_null"]["detector_config_digest"]
    )
    assert record.detector_trace["same_image_registered_wrong_reuse"] is True
    assert record.detector_trace["paired_clean_primary_null"] is True
    assert record.threshold_trace["raw_threshold_identity"] is None
    assert record.threshold_trace["rectified_threshold_identity"] is None
    assert record.module_outcome is None
    assert record.candidate_recommendation is None
    assert record.actual_elapsed_seconds > 0.0
    assert record.operation_result_payload["materialization_replay_identity"]
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
            registered_template_digest="a" * 64,
            wrong_key_template_digest="b" * 64,
            primary_null_template_digest="a" * 64,
            registered_root_key_public_digest="c" * 64,
            wrong_key_root_key_public_digest="c" * 64,
            primary_null_root_key_public_digest="c" * 64,
            registered_key_role="registered",
            wrong_key_key_role="wrong",
            primary_null_key_role="registered",
            registered_wrong_key_index=None,
            wrong_key_index=0,
            primary_null_wrong_key_index=None,
        )
        for index in range(8)
    )

    approved = evaluate_hf_transmission_direction(
        observations, budget_integrity_nonfinite_failure_count=0
    )
    blocked = evaluate_hf_transmission_direction(
        observations[:7], budget_integrity_nonfinite_failure_count=1
    )

    assert approved.registered_minus_wrong_positive_count == 7
    assert approved.registered_minus_null_positive_count == 7
    assert approved.allow_request_for_next_scientific_gate is True
    assert blocked.allow_request_for_next_scientific_gate is False


@pytest.mark.unit
def test_hf_transport_entrypoint_commits_retry_terminal_failure_and_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    environment = {
        "CEG_WM_ROOT_KEY": ROOT_KEY,
        "HF_TOKEN": "hf_transport_test_token",
    }
    retry_root = tmp_path / "retry_persistent"
    _install_entrypoint_fakes(
        monkeypatch,
        tmp_path,
        [
            _CudaDiagnosticBackend(
                failure_run_index=4,
                failure_factory=lambda: MemoryError("test resource exhaustion"),
            ),
            _CudaDiagnosticBackend(),
        ],
    )

    first_code, first_result = (
        hf_transmission_diagnostic_entrypoint.execute_hf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=retry_root,
            cache_root=tmp_path / "cache",
            run_id="hf_transmission_retry_recovery",
            session_id="hf_transport_retry_session",
            environment=environment,
        )
    )
    assert first_code == 0
    assert first_result["termination_reason"] == "retryable_resource_failure"
    assert first_result["committed_unit_count"] == 3
    assert first_result["directional_decision"] is None

    second_code, second_result = (
        hf_transmission_diagnostic_entrypoint.execute_hf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=retry_root,
            cache_root=tmp_path / "cache",
            run_id="hf_transmission_retry_recovery",
            session_id="hf_transport_recovered_session",
            environment=environment,
        )
    )
    assert second_code == 0
    assert second_result["termination_reason"] == "frozen_roster_complete"
    assert second_result["committed_unit_count"] == 11
    assert second_result["directional_decision"] is not None
    assert (
        second_result["directional_decision"][
            "budget_integrity_nonfinite_failure_count"
        ]
        == 0
    )
    protocol, manifest = load_hf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    replay_backend = FakeContentBackend()
    replay_runtime = create_runtime_adapter(replay_backend)
    replay_runtime.initialize("cpu")
    replay_adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    root_public = identify_root_key(ROOT_KEY).root_key_public_digest
    replay_runner = HfTransmissionDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=replay_adapter,
        runtime_adapter=replay_runtime,
        method_code_revision=revision,
        run_id="hf_transmission_retry_recovery",
        registered_root_key=ROOT_KEY,
        root_key_public_digest=root_public,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest=canonical_digest(
            {
                "protocol_digest": protocol.digest(),
                "manifest_digest": manifest.digest(),
                "run_id": "hf_transmission_retry_recovery",
                "root_key_public_digest": root_public,
            }
        ),
        candidate_config_digest=second_result["candidate_config_digest"],
    )
    evidence = _load_terminal_scientific_evidence(
        retry_root / "hf_transmission_retry_recovery"
    )
    assert len(evidence) == 8
    assert replay_runner.replay_directional_decision(evidence).cluster_count == 8
    with pytest.raises(HfTransmissionRunnerError, match="coverage is incomplete"):
        replay_runner.replay_directional_decision(evidence[:-1])
    with pytest.raises(HfTransmissionRunnerError, match="indexes drifted"):
        replay_runner.replay_directional_decision(evidence[:-1] + (evidence[-2],))
    with pytest.raises(HfTransmissionRunnerError, match="binding drifted"):
        replay_runner.replay_directional_decision(
            ((evidence[0][0], evidence[1][1]),) + evidence[1:]
        )
    replay_runtime.close()

    terminal_root = tmp_path / "terminal_persistent"
    _install_entrypoint_fakes(
        monkeypatch,
        tmp_path,
        [
            _CudaDiagnosticBackend(
                failure_run_index=4,
                failure_factory=lambda: ValueError("test implementation failure"),
            ),
            _CudaDiagnosticBackend(),
        ],
    )
    terminal_code, terminal_result = (
        hf_transmission_diagnostic_entrypoint.execute_hf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=terminal_root,
            cache_root=tmp_path / "cache",
            run_id="hf_transmission_terminal_failure",
            session_id="hf_transport_terminal_session",
            environment=environment,
        )
    )
    assert terminal_code == 0
    assert terminal_result["termination_reason"] == "terminal_scientific_failure"
    assert terminal_result["committed_unit_count"] == 3
    assert terminal_result["directional_decision"] is None

    resumed_code, resumed_result = (
        hf_transmission_diagnostic_entrypoint.execute_hf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=terminal_root,
            cache_root=tmp_path / "cache",
            run_id="hf_transmission_terminal_failure",
            session_id="hf_transport_terminal_resume",
            environment=environment,
        )
    )
    assert resumed_code == 0
    assert resumed_result["termination_reason"] == "frozen_roster_complete"
    assert resumed_result["committed_unit_count"] == 10
    assert resumed_result["directional_decision"] is not None
    assert (
        resumed_result["directional_decision"][
            "budget_integrity_nonfinite_failure_count"
        ]
        == 1
    )
    assert (
        resumed_result["directional_decision"][
            "allow_request_for_next_scientific_gate"
        ]
        is False
    )
    marker_paths = tuple(
        (terminal_root / "hf_transmission_terminal_failure" / "markers").glob(
            "*.COMMITTED.json"
        )
    )
    assert len(marker_paths) == 10
