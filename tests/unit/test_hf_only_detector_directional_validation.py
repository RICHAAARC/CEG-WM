"""CPU contracts for HF-only detector directional validation."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
from hashlib import sha256
import json
from struct import pack, unpack
from zipfile import ZipFile

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
    paired_rgb8_quality,
)
from experiments.protocol.development_records import DevelopmentOperationalRecord
from experiments.protocol.hf_only_detector_directional_validation import (
    HfDetectorDirectionalProtocolError,
    PriorDevelopmentManifestBinding,
    canonical_digest,
    load_authority_deny_axes,
    load_hf_only_detector_directional_protocol,
)
from experiments.runners.hf_only_detector_directional_validation import (
    HfOnlyDetectorDirectionalRunner,
)
from main import derive_wrong_key_material, identify_root_key
from main.content_chain.hf_detector import OBSERVATION_PROTOCOL
from runtime import RuntimeDeviceCapabilities, Sd35RuntimeAdapter, create_runtime_adapter
from scripts.experiment_execution import (
    hf_only_detector_directional_validation_entrypoint as directional_entrypoint,
    hf_only_detector_directional_validation_server as directional_server,
)
from scripts.experiment_execution.hf_only_detector_directional_validation_entrypoint import (
    _derive_registered_experiment_root,
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
        callbacks = tuple(tuple(range(20)) for _ in range(80))
        super().__init__(callback_sequences=callbacks)  # type: ignore[arg-type]

    def probe_devices(self) -> RuntimeDeviceCapabilities:
        return RuntimeDeviceCapabilities(cpu_available=True, cuda_device_count=1)

    def set_development_generation_prompts(self, _prompt: str) -> None:
        return None

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = super().vae_decode(latent)
        return decoded.repeat(1, 3, 1, 1)


class _RetryDirectionalCudaBackend(_DirectionalCudaBackend):
    def __init__(self) -> None:
        super().__init__()
        self.decode_call_count = 0

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.decode_call_count += 1
        if self.decode_call_count in {5, 6}:
            raise MemoryError("controlled directional resource exhaustion")
        return super().vae_decode(latent)


def _runner() -> tuple[HfOnlyDetectorDirectionalRunner, Sd35RuntimeAdapter]:
    protocol, manifest = load_hf_only_detector_directional_protocol(
        PROTOCOL, repository_root=ROOT
    )
    runtime = create_runtime_adapter(_DirectionalCudaBackend())
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
    )
    root_public = identify_root_key(registered_root).root_key_public_digest
    return (
        HfOnlyDetectorDirectionalRunner(
            protocol=protocol,
            manifest=manifest,
            adapter=adapter,
            runtime_adapter=runtime,
            method_code_revision="a" * 40,
            run_id="hf_detector_directional_test",
            registered_root_key=registered_root,
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
    assert {item.path for item in protocol.prior_development_manifests} == {
        "configs/experiments/hf_transmission_diagnostic_manifest.json",
        "configs/experiments/development_exploration_prompt_roster.json",
        "configs/experiments/thirteen_module_mechanism_screening_prompt_roster.json",
        "configs/experiments/hf_only_reference_prompt_roster.json",
        "configs/experiments/hf_only_content_threshold_fit_manifest.json",
        "configs/experiments/hf_only_untouched_confirmation_manifest.json",
    }
    deny_axes = load_authority_deny_axes(protocol.prior_development_manifests, ROOT)
    assert not {item.prompt_digest for item in manifest.entries} & set(
        deny_axes.prompt_digests
    )
    assert not {
        manifest.source_cluster_namespace,
        *(item.cluster_identity for item in manifest.entries),
    } & set(deny_axes.source_cluster_identities)
    assert manifest.seed_namespace not in deny_axes.seed_namespaces
    assert not {item.generation_seed for item in manifest.entries} & set(
        deny_axes.generation_seeds
    )
    assert not {
        manifest.image_lineage_namespace,
        *(item.image_lineage_identity for item in manifest.entries),
    } & set(deny_axes.image_lineage_identities)
    assert not {
        manifest.registered_key_derivation_identity,
        manifest.wrong_key_control_identity,
    } & set(deny_axes.key_control_identities)


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
    assert scientific.detector_trace["public_callable"] == "main.hf_detector"
    assert scientific.detector_trace["same_image_registered_wrong_reuse"] is True
    assert scientific.detector_trace["paired_clean_primary_null"] is True
    assert scientific.detector_trace["reference_image_used"] is False
    assert scientific.detector_trace["embed_record_used"] is False
    assert scientific.detector_trace["private_latent_used_by_detector"] is False
    assert scientific.threshold_trace["raw_threshold_identity"] is None
    assert scientific.module_outcome is None
    assert scientific.candidate_recommendation is None
    assert observation["rgb_quality_dtype"] == "torch.uint8"
    assert observation["rgb_paired_mse"] >= 0.0
    assert observation["actual_runtime_dtype"] == "torch.float16"


@pytest.mark.unit
def test_hf_detector_directional_runner_accepts_embedder_binary32_budget_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, runtime = _runner()
    runtime_result = runner._execute_paired_runtime(_base_latent())
    binary32_limit = unpack(">f", pack(">f", 3 / 250))[0]
    assert binary32_limit > 3 / 250
    boundary_result = replace(
        runtime_result,
        content_materialization=replace(
            runtime_result.content_materialization,
            realized_relative_l2=binary32_limit,
        ),
        content_materialization_result=replace(
            runtime_result.content_materialization_result,
            content_relative_l2_nominal=binary32_limit,
            content_relative_l2_limit=binary32_limit,
            realized_relative_l2=binary32_limit,
        ),
    )
    monkeypatch.setattr(
        runner, "_execute_paired_runtime", lambda _base_latent: boundary_result
    )

    record = runner.execute_scientific_cluster(
        cluster_ordinal=0,
        base_latent=_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    runtime.close()

    assert record.execution_status == "success"
    assert record.operation_result_payload["realized_relative_l2"] == binary32_limit
    assert record.operation_result_payload["content_relative_l2_limit"] == binary32_limit


@pytest.mark.unit
def test_hf_detector_directional_entrypoint_commits_controlled_evidence_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "directional_failure_package.zip"
    package.write_bytes(b"directional-failure-package")
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

    original_execute = Sd35RuntimeAdapter.execute_content_write_and_vae
    execution_count = 0
    binary32_limit = unpack(">f", pack(">f", 3 / 250))[0]

    def execute_with_controlled_evidence_failure(
        runtime_adapter: Sd35RuntimeAdapter,
        base_latent: torch.Tensor,
        content_embedding_operation,
    ):
        nonlocal execution_count
        execution_count += 1
        result = original_execute(
            runtime_adapter, base_latent, content_embedding_operation
        )
        if execution_count != 3:
            return result
        return replace(
            result,
            content_materialization=replace(
                result.content_materialization,
                realized_relative_l2=binary32_limit,
                integrity_status="write_disappeared",
            ),
            content_materialization_result=replace(
                result.content_materialization_result,
                content_relative_l2_nominal=binary32_limit,
                content_relative_l2_limit=binary32_limit + 0.001,
                realized_relative_l2=binary32_limit,
            ),
        )

    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "execute_content_write_and_vae",
        execute_with_controlled_evidence_failure,
    )

    exit_code, result = (
        directional_entrypoint.execute_hf_only_detector_directional_validation_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="hf_detector_directional_controlled_failure_test",
            session_id="directional_controlled_failure_session",
            environment={"HF_TOKEN": "test-token", "CEG_WM_ROOT_KEY": ROOT_KEY},
            authorized_scientific_unit_count=8,
        )
    )
    assert exit_code == 0
    assert result["committed_unit_count"] == 10

    marker_root = (
        tmp_path
        / "persistent"
        / "hf_detector_directional_controlled_failure_test"
        / "markers"
    )
    markers = tuple(
        json.loads(path.read_text("utf-8"))
        for path in marker_root.glob("*.COMMITTED.json")
    )
    scientific_failure_markers = tuple(
        marker for marker in markers if marker["unit_index"] == 2
    )
    assert len(scientific_failure_markers) == 1
    marker = scientific_failure_markers[0]
    bundle = (
        tmp_path
        / "persistent"
        / "hf_detector_directional_controlled_failure_test"
        / "bundles"
        / f"sha256_{marker['bundle_sha256']}.zip"
    )
    with ZipFile(bundle) as archive:
        record_member = next(
            name for name in archive.namelist() if name != "artifact_manifest.json"
        )
        record_payload = json.loads(archive.read(record_member))
    assert record_payload["execution_status"] == "failed"
    assert record_payload["operation_result_payload"] == {
        "failure_stage": "hf_detector_directional_runtime_operation",
        "failure_type": "experiments.runners.hf_only_detector_directional_validation.HfDetectorDirectionalEvidenceViolation",
        "result_available": False,
        "failure_category": "budget_violation",
        "realized_relative_l2": binary32_limit,
        "content_relative_l2_limit": binary32_limit + 0.001,
        "budget_status": "accepted",
        "integrity_status": "write_disappeared",
    }


@pytest.mark.unit
def test_hf_detector_directional_rgb_quality_requires_actual_uint8_values() -> None:
    relative_l2, mse = paired_rgb8_quality((0, 255), (0, 254))
    assert relative_l2 == pytest.approx(1 / 254)
    assert mse == pytest.approx(0.5)
    with pytest.raises(HfDetectorDirectionalMetricError, match="RGB values"):
        paired_rgb8_quality((0.0, 255.0), (0.0, 254.0))  # type: ignore[arg-type]


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
        registered_detector_identity="shared_hf_detector_operation",
        wrong_key_detector_identity="shared_hf_detector_operation",
        primary_null_detector_identity="shared_hf_detector_operation",
        detector_config_digest="d" * 64,
        observation_protocol=OBSERVATION_PROTOCOL,
        detector_statistic_identity="hf_direct_score_centered_normalized_correlation",
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
        rgb_paired_mse=1.0,
        rgb_quality_dtype="torch.uint8",
        actual_runtime_dtype="torch.float16",
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
def test_hf_detector_directional_floor_is_strict_and_failures_keep_frozen_denominator() -> None:
    equality = create_hf_detector_directional_observation(
        **{
            **{
                field: getattr(_observation(0), field)
                for field in _observation(0).__dataclass_fields__
                if field not in {
                    "registered_score", "registered_minus_wrong_key",
                    "registered_minus_primary_null", "observation_identity",
                    "registered_minus_wrong_key_strict_floor_passed",
                    "registered_minus_wrong_key_exact_tie",
                    "registered_minus_primary_null_strict_floor_passed",
                    "registered_minus_primary_null_exact_tie",
                }
            },
            "registered_score": 0.001,
        }
    )
    assert equality.registered_minus_wrong_key_strict_floor_passed is False
    assert equality.registered_minus_primary_null_strict_floor_passed is False
    all_failed = aggregate_hf_detector_direction((), failed_cluster_count=32)
    assert all_failed.registered_minus_wrong_key.observation_count == 32
    assert all_failed.registered_minus_wrong_key.practical_success_count == 0
    assert all_failed.registered_minus_wrong_key.threshold_free_paired_ranking_auc == 0.0
    assert all_failed.registered_minus_wrong_key.mean_margin is None
    assert all_failed.allow_request_for_next_scientific_gate is False


@pytest.mark.unit
def test_hf_detector_directional_rejects_mixed_detector_identity() -> None:
    source = _observation(0)
    with pytest.raises(HfDetectorDirectionalMetricError, match="control binding"):
        replace(source, wrong_key_detector_identity="different_detector").validate()


@pytest.mark.unit
def test_hf_detector_directional_registered_subdomain_differs_from_base_and_wrong_control() -> None:
    protocol, manifest = load_hf_only_detector_directional_protocol(
        PROTOCOL, repository_root=ROOT
    )
    registered = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
    )
    assert registered == _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
    )
    assert registered.startswith("ceg-wm-hf-directional-registered-v2:")
    assert registered != _derive_registered_experiment_root(
        ROOT_KEY + "-other",
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
    )
    assert registered != _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest() + "-other",
        manifest_digest=manifest.digest(),
    )
    assert registered != _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest() + "-other",
    )
    base_digest = identify_root_key(ROOT_KEY).root_key_public_digest
    registered_digest = identify_root_key(registered).root_key_public_digest
    assert registered_digest != base_digest
    assert derive_wrong_key_material(registered_digest, 0).material_text != derive_wrong_key_material(base_digest, 0).material_text


@pytest.mark.unit
def test_hf_detector_directional_authority_files_fail_closed_when_missing(tmp_path: Path) -> None:
    binding = PriorDevelopmentManifestBinding(
        path="configs/experiments/missing_authority.json",
        file_sha256="0" * 64,
    )
    with pytest.raises(HfDetectorDirectionalProtocolError, match="missing or unreadable"):
        load_authority_deny_axes((binding,), tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overlap_payload",
    [
        {"prompt": "an amber lantern glowing beside a quiet stone path"},
        {"source_cluster_id": "scientific_amber_lantern"},
        {"seed_namespace": "hf_only_detector_directional_validation_20260808"},
        {"generation_seed": 202608081100},
        {"image_lineage_identity": "paired_clean_hf_rendered_rgb8_lineage"},
        {"key_control_identity": "hf_detector_directional_wrong_key_control_roster"},
    ],
)
def test_hf_detector_directional_rejects_each_authority_axis_independently(
    tmp_path: Path, overlap_payload: dict[str, object]
) -> None:
    raw = json.loads(PROTOCOL.read_text("utf-8"))
    repository = tmp_path / "repository"
    for item in raw["prior_development_manifests"]:
        source = ROOT / item["path"]
        target = repository / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    manifest_target = repository / raw["manifest_path"]
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_bytes((ROOT / raw["manifest_path"]).read_bytes())
    overlap_path = repository / "configs/experiments/directional_axis_overlap.json"
    overlap_path.write_text(json.dumps(overlap_payload), "utf-8")
    raw["prior_development_manifests"].append(
        {
            "path": "configs/experiments/directional_axis_overlap.json",
            "file_sha256": sha256(overlap_path.read_bytes()).hexdigest(),
        }
    )
    bindings = tuple(
        PriorDevelopmentManifestBinding(**item)
        for item in raw["prior_development_manifests"]
    )
    axes = load_authority_deny_axes(bindings, repository)
    raw["source_cluster_deny_list_digest"] = canonical_digest(
        {
            "manifest_bindings": tuple(asdict(item) for item in bindings),
            "authority_deny_axes": axes.digest_value(),
        }
    )
    protocol_target = repository / "configs/experiments/hf_only_detector_directional_validation.json"
    protocol_target.write_text(json.dumps(raw), "utf-8")
    with pytest.raises(HfDetectorDirectionalProtocolError, match="authority axis"):
        load_hf_only_detector_directional_protocol(
            protocol_target, repository_root=repository
        )


@pytest.mark.unit
@pytest.mark.parametrize("count", [1, 7, 9, 31])
def test_hf_detector_directional_server_rejects_non_boundary_science_counts(
    tmp_path: Path, count: int
) -> None:
    with pytest.raises(directional_server.HfDetectorDirectionalServerError, match="first gate or full roster"):
        directional_server.execute_hf_only_detector_directional_validation_server_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="hf_directional_boundary_test",
            session_id="hf_directional_boundary_session",
            environment={"HF_TOKEN": "token", "CEG_WM_ROOT_KEY": ROOT_KEY},
            install_dependencies=False,
            authorized_scientific_unit_count=count,
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

    with pytest.raises(
        directional_entrypoint.HfDetectorDirectionalEntrypointError,
        match="requires verified first-gate",
    ):
        directional_entrypoint.execute_hf_only_detector_directional_validation_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "fresh_full_persistent",
            cache_root=tmp_path / "fresh_full_cache",
            run_id="hf_detector_directional_fresh_full_test",
            session_id="directional_fresh_full_session",
            environment={"HF_TOKEN": "test-token", "CEG_WM_ROOT_KEY": ROOT_KEY},
            authorized_scientific_unit_count=32,
        )

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

    continuation_code, continuation = (
        directional_entrypoint.execute_hf_only_detector_directional_validation_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id="hf_detector_directional_entrypoint_test",
            session_id="directional_continuation_session",
            environment={"HF_TOKEN": "test-token", "CEG_WM_ROOT_KEY": ROOT_KEY},
            authorized_scientific_unit_count=32,
        )
    )
    assert continuation_code == 0
    assert continuation["termination_reason"] == "frozen_roster_complete"
    assert continuation["committed_unit_count"] == 34
    assert continuation["directional_aggregate"]["expected_cluster_count"] == 32


@pytest.mark.unit
def test_hf_detector_directional_entrypoint_recovers_resource_attempt_before_terminal_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "retry_directional_package.zip"
    package.write_bytes(b"retry-directional-package")
    backend = _RetryDirectionalCudaBackend()
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
    common = {
        "repository_root": ROOT,
        "expected_revision": "a" * 40,
        "persistent_root": tmp_path / "retry_persistent",
        "cache_root": tmp_path / "retry_cache",
        "run_id": "hf_detector_directional_resource_retry_test",
        "environment": {"HF_TOKEN": "test-token", "CEG_WM_ROOT_KEY": ROOT_KEY},
    }

    first_code, first = directional_entrypoint.execute_hf_only_detector_directional_validation_session(
        **common,
        session_id="directional_resource_attempt_initial",
        authorized_scientific_unit_count=8,
    )
    assert first_code == 0
    assert first["termination_reason"] == "retryable_resource_failure_after_committed_attempt"
    assert first["committed_unit_count"] == 3

    second_code, second = directional_entrypoint.execute_hf_only_detector_directional_validation_session(
        **common,
        session_id="directional_resource_attempt_terminal",
        authorized_scientific_unit_count=8,
    )
    assert second_code == 0
    assert second["termination_reason"] == "terminal_resource_failure_after_committed_attempt"
    assert second["committed_unit_count"] == 4

    markers = []
    for path in sorted((tmp_path / "retry_persistent" / common["run_id"] / "markers").glob("*.json")):
        markers.append(json.loads(path.read_text("utf-8")))
    attempts = sorted(
        (item for item in markers if item["unit_index"] == 2),
        key=lambda item: item["attempt_index"],
    )
    assert [item["attempt_disposition"] for item in attempts] == [
        "retryable_resource_failure",
        "final_failure",
    ]
    assert attempts[1]["parent_attempt_intent_digest"] == attempts[0]["intent_digest"]
    assert attempts[1]["attempt_index"] == 1
    attempt_records = []
    bundle_root = tmp_path / "retry_persistent" / common["run_id"] / "bundles"
    for marker in attempts:
        with ZipFile(bundle_root / f"sha256_{marker['bundle_sha256']}.zip") as archive:
            record_member = next(
                name
                for name in archive.namelist()
                if name != "artifact_manifest.json"
            )
            attempt_records.append(json.loads(archive.read(record_member)))
    assert [item["execution_status"] for item in attempt_records] == [
        "retry",
        "failed",
    ]
    assert attempt_records[0]["failure_class"] == "resource_failure"
    assert attempt_records[1]["failure_class"] == "resource_failure"
    assert attempt_records[1]["retry_parent_intent_digest"] == attempts[0]["intent_digest"]

    gate_code, gate = directional_entrypoint.execute_hf_only_detector_directional_validation_session(
        **common,
        session_id="directional_resource_attempt_gate_completion",
        authorized_scientific_unit_count=8,
    )
    assert gate_code == 0
    assert gate["termination_reason"] == "authorized_directional_unit_boundary_reached"
    assert gate["committed_unit_count"] == 11

    final_code, final = directional_entrypoint.execute_hf_only_detector_directional_validation_session(
        **common,
        session_id="directional_resource_attempt_complete",
        authorized_scientific_unit_count=32,
    )
    assert final_code == 0
    assert final["termination_reason"] == "frozen_roster_complete"
    assert final["committed_unit_count"] == 35
    aggregate = final["directional_aggregate"]
    assert aggregate["expected_cluster_count"] == 32
    assert aggregate["successful_cluster_count"] == 31
    assert aggregate["failed_cluster_count"] == 1
    assert aggregate["registered_minus_primary_null"]["observation_count"] == 32
    assert aggregate["registered_minus_wrong_key"]["observation_count"] == 32
    assert aggregate["allow_request_for_next_scientific_gate"] is False
