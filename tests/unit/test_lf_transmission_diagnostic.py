"""CPU contracts for the real LF transport diagnostic execution path."""

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
from experiments.metrics.lf_transmission_diagnostic import (
    LfTransmissionMetricError,
    create_lf_signal_position_observation,
    evaluate_lf_transmission_direction,
)
from experiments.protocol.lf_transmission_diagnostic import (
    RUN_ID,
    SIGNAL_POSITIONS,
    canonical_digest,
    derive_lf_transmission_analysis_identity,
    load_lf_transmission_protocol,
)
from experiments.protocol.hf_transmission_diagnostic import (
    canonical_digest as canonical_hf_transmission_digest,
    load_hf_transmission_manifest,
)
from experiments.protocol.hf_only_detector_directional_validation import (
    canonical_digest as canonical_hf_directional_digest,
    load_hf_detector_directional_manifest,
)
from experiments.protocol.hf_only_reference_protocol import (
    load_compact_hf_only_reference_split_manifest,
    load_frozen_prompt_roster,
    materialize_hf_only_reference_split_manifest,
)
from experiments.protocol.internal_splits import derive_source_cluster_id
from experiments.protocol.development_records import DevelopmentScientificRecord
from experiments.runners.development_persistence import CommittedUnit
from experiments.runners.lf_transmission_diagnostic import (
    LfTransmissionDiagnosticRunner,
    LfTransmissionRunnerError,
)
from main import identify_root_key
from runtime import create_runtime_adapter
from runtime import RuntimeDeviceCapabilities, Sd35RuntimeAdapter
from scripts.experiment_execution import lf_transmission_diagnostic_entrypoint
from scripts.experiment_execution.lf_transmission_diagnostic_entrypoint import (
    _derive_registered_experiment_root,
)
from tests.unit.test_runtime_content_write_and_vae import (
    FakeContentBackend,
    _base_latent as small_base_latent,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/lf_transmission_diagnostic.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-lf-transmission-diagnostic-test-key"


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
    package = tmp_path / "lf_transmission_test_package.zip"
    package.write_bytes(b"checked-in-test-package")
    monkeypatch.setattr(
        lf_transmission_diagnostic_entrypoint,
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
        lf_transmission_diagnostic_entrypoint,
        "_build_or_verify_package",
        lambda *_args: package,
    )
    monkeypatch.setattr(
        lf_transmission_diagnostic_entrypoint,
        "_base_latent",
        lambda generation_seed, **_kwargs: (
            small_base_latent().to(torch.float32)
            + float(generation_seed % 97) / 10000.0
        ).to(torch.float16),
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
def test_lf_transmission_protocol_freezes_isolated_eight_cluster_roster() -> None:
    protocol, manifest = load_lf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )

    assert protocol.split == "development"
    assert protocol.run_id == RUN_ID
    assert protocol.role_id == "lf_transmission_diagnostic"
    assert len(manifest.entries) == 8
    assert {item.role_id for item in manifest.entries} == {
        "lf_transmission_diagnostic"
    }
    assert protocol.operational_unit_count == 0
    assert len(protocol.unit_roster) == 8
    assert tuple(item.unit_index for item in protocol.unit_roster) == tuple(range(8))
    assert all(
        item.phase == "development_scientific_breadth"
        for item in protocol.unit_roster
    )
    assert sum(item.responsibility_id == "lf_detector" for item in protocol.unit_roster) == 8
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
        registered_key_derivation_identity=(
            manifest.registered_key_derivation_identity
        ),
        registered_key_domain_identity=manifest.registered_key_domain_identity,
        registered_key_family_namespace=manifest.registered_key_family_namespace,
    )
    root_public = identify_root_key(registered_root).root_key_public_digest
    lf_identities = {
        derive_lf_transmission_analysis_identity(
            item,
            manifest,
            root_key_public_digest=root_public,
        )
        for item in manifest.entries
    }
    hf_manifest = load_hf_transmission_manifest(
        ROOT / "configs/experiments/hf_transmission_diagnostic_manifest.json"
    )
    directional_manifest = load_hf_detector_directional_manifest(
        ROOT
        / "configs/experiments/hf_only_detector_directional_validation_manifest.json"
    )
    prompt_roster = load_frozen_prompt_roster(
        ROOT / "configs/experiments/hf_only_reference_prompt_roster.json"
    )
    compact_manifests = tuple(
        load_compact_hf_only_reference_split_manifest(ROOT / path)
        for path in (
            "configs/experiments/hf_only_content_threshold_fit_manifest.json",
            "configs/experiments/hf_only_untouched_confirmation_manifest.json",
        )
    )
    formal_manifests = tuple(
        materialize_hf_only_reference_split_manifest(compact, prompt_roster)
        for compact in compact_manifests
    )
    hf_transport_key_family = canonical_hf_transmission_digest(
        {
            "root_key_public_digest": root_public,
            "seed_namespace": hf_manifest.seed_namespace,
            "role": "registered_hf_transmission_detection_key_family",
        }
    )
    hf_transport_cluster_ids = {
        derive_source_cluster_id(
            prompt_digest=item.prompt_digest,
            generation_seed=item.generation_seed,
            image_lineage_digest=item.image_lineage_digest,
            registered_key_family_digest=hf_transport_key_family,
        )
        for item in hf_manifest.entries
    }
    directional_key_family = canonical_hf_directional_digest(
        {
            "root_key_public_digest": root_public,
            "seed_namespace": directional_manifest.seed_namespace,
            "role": "registered_hf_detector_directional_key_family",
        }
    )
    directional_cluster_ids = {
        derive_source_cluster_id(
            prompt_digest=item.prompt_digest,
            generation_seed=item.generation_seed,
            image_lineage_digest=item.image_lineage_digest,
            registered_key_family_digest=directional_key_family,
        )
        for item in directional_manifest.entries
    }
    formal_assignments = tuple(
        assignment
        for formal_manifest in formal_manifests
        for assignment in formal_manifest.assignments
    )
    existing_prompts = {
        *(item.prompt for item in hf_manifest.entries),
        *(item.prompt for item in directional_manifest.entries),
        *(item.prompt_text for item in prompt_roster.rows),
    }
    existing_prompt_digests = {
        *(item.prompt_digest for item in hf_manifest.entries),
        *(item.prompt_digest for item in directional_manifest.entries),
        *(item.prompt_digest for item in prompt_roster.rows),
    }
    existing_generation_seeds = {
        *(item.generation_seed for item in hf_manifest.entries),
        *(item.generation_seed for item in directional_manifest.entries),
        *(item.identity.generation_seed for item in formal_assignments),
    }
    existing_source_clusters = {
        *hf_transport_cluster_ids,
        *directional_cluster_ids,
        *(item.identity.source_cluster_id for item in formal_assignments),
    }
    existing_lineage_digests = {
        *(item.image_lineage_digest for item in hf_manifest.entries),
        *(item.image_lineage_digest for item in directional_manifest.entries),
        *(item.identity.image_lineage_digest for item in formal_assignments),
    }

    assert {item.prompt for item in manifest.entries}.isdisjoint(existing_prompts)
    assert {item.prompt_digest for item in manifest.entries}.isdisjoint(
        existing_prompt_digests
    )
    assert {item.generation_seed for item in manifest.entries}.isdisjoint(
        existing_generation_seeds
    )
    assert {item.cluster_identity for item in manifest.entries}.isdisjoint(
        {
            *(item.cluster_identity for item in hf_manifest.entries),
            *(item.cluster_identity for item in directional_manifest.entries),
        }
    )
    assert {item.source_cluster_id for item in lf_identities}.isdisjoint(
        existing_source_clusters
    )
    assert {item.image_lineage_digest for item in manifest.entries}.isdisjoint(
        existing_lineage_digests
    )
    assert manifest.seed_namespace not in {
        hf_manifest.seed_namespace,
        directional_manifest.seed_namespace,
        *(item.seed_namespace for item in compact_manifests),
    }
    assert manifest.source_cluster_namespace != (
        directional_manifest.source_cluster_namespace
    )
    assert manifest.image_lineage_namespace != (
        directional_manifest.image_lineage_namespace
    )
    assert manifest.registered_key_derivation_identity != (
        directional_manifest.registered_key_derivation_identity
    )
    assert manifest.registered_key_family_namespace not in {
        hf_transport_key_family,
        directional_key_family,
        *(item.registered_key_family_digest for item in compact_manifests),
    }
    assert {item.registered_key_family_digest for item in lf_identities}.isdisjoint(
        {
            hf_transport_key_family,
            directional_key_family,
            *(item.registered_key_family_digest for item in compact_manifests),
        }
    )


@pytest.mark.unit
def test_lf_transport_entrypoint_uses_derived_registered_root_for_public_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, manifest = load_lf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    derived_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
        registered_key_derivation_identity=(
            manifest.registered_key_derivation_identity
        ),
        registered_key_domain_identity=manifest.registered_key_domain_identity,
        registered_key_family_namespace=manifest.registered_key_family_namespace,
    )
    derive = lambda **overrides: _derive_registered_experiment_root(
        overrides.get("base_root_key", ROOT_KEY),
        protocol_digest=overrides.get("protocol_digest", protocol.digest()),
        manifest_digest=overrides.get("manifest_digest", manifest.digest()),
        registered_key_derivation_identity=overrides.get(
            "registered_key_derivation_identity",
            manifest.registered_key_derivation_identity,
        ),
        registered_key_domain_identity=overrides.get(
            "registered_key_domain_identity",
            manifest.registered_key_domain_identity,
        ),
        registered_key_family_namespace=overrides.get(
            "registered_key_family_namespace",
            manifest.registered_key_family_namespace,
        ),
    )
    assert derived_root == derive()
    assert derived_root.startswith("ceg-wm-lf-transmission-registered-v2:")
    for field in (
        "base_root_key",
        "protocol_digest",
        "manifest_digest",
        "registered_key_derivation_identity",
        "registered_key_domain_identity",
        "registered_key_family_namespace",
    ):
        assert derived_root != derive(**{field: "v2-authority-change"})
    assert identify_root_key(derived_root).root_key_public_digest != (
        identify_root_key(ROOT_KEY).root_key_public_digest
    )

    public_call_keys: list[object] = []
    build_lf_carrier = CegWmExperimentAdapter.build_lf_carrier
    detect_lf = CegWmExperimentAdapter.detect_lf

    def capture_build_lf_carrier(self, detection_key, shape, **kwargs):
        public_call_keys.append(detection_key)
        return build_lf_carrier(self, detection_key, shape, **kwargs)

    def capture_detect_lf(self, observation, detection_key):
        public_call_keys.append(detection_key)
        return detect_lf(self, observation, detection_key)

    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "build_lf_carrier",
        capture_build_lf_carrier,
    )
    monkeypatch.setattr(CegWmExperimentAdapter, "detect_lf", capture_detect_lf)
    _install_entrypoint_fakes(
        monkeypatch,
        tmp_path,
        [_CudaDiagnosticBackend()],
    )

    code, result = (
        lf_transmission_diagnostic_entrypoint.execute_lf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "derived_root_persistent",
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_transport_derived_root_session",
            environment={
                "CEG_WM_ROOT_KEY": ROOT_KEY,
                "HF_TOKEN": "lf_transport_test_token",
            },
        )
    )

    assert code == 0
    assert result["termination_reason"] == "frozen_roster_complete"
    assert derived_root in public_call_keys
    assert ROOT_KEY not in public_call_keys


@pytest.mark.unit
def test_lf_transport_runner_separates_latent_diagnostics_from_formal_detector() -> None:
    protocol, manifest = load_lf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    backend = FakeContentBackend()
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
        registered_key_derivation_identity=(
            manifest.registered_key_derivation_identity
        ),
        registered_key_domain_identity=manifest.registered_key_domain_identity,
        registered_key_family_namespace=manifest.registered_key_family_namespace,
    )
    root_public = identify_root_key(registered_root).root_key_public_digest
    runner = LfTransmissionDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=adapter,
        runtime_adapter=runtime,
        method_code_revision="a" * 40,
        run_id=RUN_ID,
        registered_root_key=registered_root,
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
    assert observations[-1]["statistic_role"] == "formal_lf_detector_operation"
    assert observations[-1]["registered_template_digest"] != observations[-1][
        "wrong_key_template_digest"
    ]
    assert observations[-1]["registered_key_role"] == "registered"
    assert observations[-1]["wrong_key_key_role"] == "wrong"
    assert observations[-1]["wrong_key_index"] == 0
    formal_results = record.operation_result_payload["formal_lf_detection_results"]
    assert formal_results["registered"]["candidate_id"] == "lf_low_pass"
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
    assert record.operation_result_payload["content_materialization_result"][
        "budget_status"
    ] == "accepted"
    assert record.operation_result_payload["content_materialization_result"][
        "integrity_status"
    ] == "passed"
    runtime.close()


@pytest.mark.unit
def test_lf_transport_directional_rule_is_exactly_seven_of_eight() -> None:
    observations = tuple(
        create_lf_signal_position_observation(
            position_id="rgb_vae_reencoded",
            statistic_role="formal_lf_detector_operation",
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

    approved = evaluate_lf_transmission_direction(
        observations, budget_integrity_nonfinite_failure_count=0
    )
    blocked = evaluate_lf_transmission_direction(
        observations[:7], budget_integrity_nonfinite_failure_count=1
    )

    assert approved.registered_minus_wrong_positive_count == 7
    assert approved.registered_minus_null_positive_count == 7
    assert approved.allow_request_for_lf_directional_validation is True
    assert blocked.allow_request_for_lf_directional_validation is False
    with pytest.raises(LfTransmissionMetricError, match="duplicated"):
        evaluate_lf_transmission_direction(
            (observations[0],) * 8,
            budget_integrity_nonfinite_failure_count=0,
        )


@pytest.mark.unit
def test_lf_transport_entrypoint_commits_retry_terminal_failure_and_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    environment = {
        "CEG_WM_ROOT_KEY": ROOT_KEY,
        "HF_TOKEN": "lf_transport_test_token",
    }
    retry_root = tmp_path / "retry_persistent"
    _install_entrypoint_fakes(
        monkeypatch,
        tmp_path,
        [
            _CudaDiagnosticBackend(
                failure_run_index=0,
                failure_factory=lambda: MemoryError("test resource exhaustion"),
            ),
            _CudaDiagnosticBackend(
                failure_run_index=0,
                failure_factory=lambda: MemoryError(
                    "test repeated resource exhaustion"
                ),
            ),
            _CudaDiagnosticBackend(),
        ],
    )

    first_code, first_result = (
        lf_transmission_diagnostic_entrypoint.execute_lf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=retry_root,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_transport_retry_session",
            environment=environment,
        )
    )
    assert first_code == 0
    assert first_result["termination_reason"] == "retryable_resource_failure"
    assert first_result["committed_unit_count"] == 1
    assert first_result["directional_decision"] is None

    second_code, second_result = (
        lf_transmission_diagnostic_entrypoint.execute_lf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=retry_root,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_transport_recovered_session",
            environment=environment,
        )
    )
    assert second_code == 0
    assert second_result["termination_reason"] == "terminal_scientific_failure"
    assert second_result["committed_unit_count"] == 2
    assert second_result["directional_decision"] is None

    third_code, third_result = (
        lf_transmission_diagnostic_entrypoint.execute_lf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=retry_root,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_transport_after_exhausted_retry",
            environment=environment,
        )
    )
    assert third_code == 0
    assert third_result["termination_reason"] == "frozen_roster_complete"
    assert third_result["committed_unit_count"] == 9
    assert third_result["directional_decision"] is not None
    assert (
        third_result["directional_decision"][
            "budget_integrity_nonfinite_failure_count"
        ]
        == 1
    )
    assert (
        third_result["directional_decision"][
            "allow_request_for_lf_directional_validation"
        ]
        is False
    )
    protocol, manifest = load_lf_transmission_protocol(
        PROTOCOL, repository_root=ROOT
    )
    replay_backend = FakeContentBackend()
    replay_runtime = create_runtime_adapter(replay_backend)
    replay_runtime.initialize("cpu")
    replay_adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
        registered_key_derivation_identity=(
            manifest.registered_key_derivation_identity
        ),
        registered_key_domain_identity=manifest.registered_key_domain_identity,
        registered_key_family_namespace=manifest.registered_key_family_namespace,
    )
    root_public = identify_root_key(registered_root).root_key_public_digest
    replay_runner = LfTransmissionDiagnosticRunner(
        protocol=protocol,
        manifest=manifest,
        adapter=replay_adapter,
        runtime_adapter=replay_runtime,
        method_code_revision=revision,
        run_id=RUN_ID,
        registered_root_key=registered_root,
        root_key_public_digest=root_public,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest=canonical_digest(
            {
                "protocol_digest": protocol.digest(),
                "manifest_digest": manifest.digest(),
                "run_id": RUN_ID,
                "root_key_public_digest": root_public,
            }
        ),
        candidate_config_digest=third_result["candidate_config_digest"],
    )
    evidence = _load_terminal_scientific_evidence(
        retry_root / RUN_ID
    )
    assert len(evidence) == 8
    assert replay_runner.replay_directional_decision(evidence).cluster_count == 8
    with pytest.raises(LfTransmissionRunnerError, match="coverage is incomplete"):
        replay_runner.replay_directional_decision(evidence[:-1])
    with pytest.raises(LfTransmissionRunnerError, match="indexes drifted"):
        replay_runner.replay_directional_decision(evidence[:-1] + (evidence[-2],))
    with pytest.raises(LfTransmissionRunnerError, match="binding drifted"):
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
                failure_run_index=0,
                failure_factory=lambda: ValueError("test implementation failure"),
            ),
            _CudaDiagnosticBackend(),
        ],
    )
    terminal_code, terminal_result = (
        lf_transmission_diagnostic_entrypoint.execute_lf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=terminal_root,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_transport_terminal_session",
            environment=environment,
        )
    )
    assert terminal_code == 0
    assert terminal_result["termination_reason"] == "terminal_scientific_failure"
    assert terminal_result["committed_unit_count"] == 1
    assert terminal_result["directional_decision"] is None

    resumed_code, resumed_result = (
        lf_transmission_diagnostic_entrypoint.execute_lf_transmission_diagnostic_session(
            repository_root=ROOT,
            expected_revision=revision,
            persistent_root=terminal_root,
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_transport_terminal_resume",
            environment=environment,
        )
    )
    assert resumed_code == 0
    assert resumed_result["termination_reason"] == "frozen_roster_complete"
    assert resumed_result["committed_unit_count"] == 8
    assert resumed_result["directional_decision"] is not None
    assert (
        resumed_result["directional_decision"][
            "budget_integrity_nonfinite_failure_count"
        ]
        == 1
    )
    assert (
        resumed_result["directional_decision"][
            "allow_request_for_lf_directional_validation"
        ]
        is False
    )
    marker_paths = tuple(
        (terminal_root / RUN_ID / "markers").glob(
            "*.COMMITTED.json"
        )
    )
    assert len(marker_paths) == 8
