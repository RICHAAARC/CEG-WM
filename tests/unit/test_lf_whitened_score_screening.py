"""Protocol, fit, runner, and persistence checks for LF whitening screening."""

from __future__ import annotations

from dataclasses import asdict
import json
from math import cos, pi
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    ComponentCallObservation,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics.lf_whitened_score_screening import (
    LfWhitenedScoreMetricError,
    clean_null_band_energy_sums,
    create_lf_whitened_screening_observation,
    evaluate_lf_whitened_screening,
    fit_lf_null_whitening_asset,
)
from experiments.protocol.lf_whitened_score_screening import (
    MARGIN_FLOOR,
    RUN_ID,
    canonical_digest,
    load_lf_whitened_score_screening_protocol,
)
from experiments.protocol.development_records import DevelopmentOperationalRecord
from experiments.runners.development_persistence import (
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from experiments.runners.lf_whitened_score_screening import (
    LfWhitenedScoreScreeningRunner,
)
import experiments.runners.lf_whitened_score_screening as screening_runner_module
from main import (
    DerivedWrongKeyMaterial,
    LfDetectionResult,
    LfNullWhitenedDetectionResult,
    identify_root_key,
)
from runtime import RuntimeDeviceCapabilities, Sd35RuntimeAdapter, create_runtime_adapter
from scripts.experiment_execution import lf_whitened_score_screening_entrypoint
from scripts.experiment_execution.lf_whitened_score_screening_entrypoint import (
    _derive_registered_experiment_root,
)
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/lf_whitened_score_screening.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-lf-whitened-screening-test-key"


class _EntrypointBackend(FakeContentBackend):
    def __init__(
        self,
        *,
        callback_count: int,
        failure_run_index: int | None = None,
        failure_factory=None,
    ) -> None:
        super().__init__(
            callback_sequences=tuple(
                tuple(range(20)) for _ in range(callback_count)
            )
        )  # type: ignore[arg-type]
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


@pytest.mark.unit
def test_lf_whitened_protocol_freezes_clean_fit_before_paired_screening() -> None:
    protocol, fit, screening = load_lf_whitened_score_screening_protocol(
        PROTOCOL, repository_root=ROOT
    )

    assert protocol.run_id == RUN_ID
    assert protocol.operational_unit_count == 1
    assert len(protocol.unit_roster) == 41
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(
        range(41)
    )
    assert protocol.unit_roster[0].phase == "development_environment_preflight"
    assert (
        protocol.unit_roster[0].responsibility_id
        == "lf_clean_public_vae_runtime_preflight"
    )
    assert {
        unit.responsibility_id for unit in protocol.unit_roster[1:33]
    } == {"lf_whitening_null_fit"}
    assert {
        unit.content_branch_id for unit in protocol.unit_roster[1:33]
    } == {"clean_control"}
    assert {
        unit.responsibility_id for unit in protocol.unit_roster[33:]
    } == {"lf_whitened_score_screening"}
    assert {
        unit.content_branch_id for unit in protocol.unit_roster[33:]
    } == {"lf_only"}
    assert protocol.margin_floor == float.fromhex("0x1.0000000000000p-10")
    assert len(fit.entries) == 32
    assert len(screening.entries) == 8
    assert {item.prompt_digest for item in fit.entries}.isdisjoint(
        {item.prompt_digest for item in screening.entries}
    )
    assert {item.generation_seed for item in fit.entries}.isdisjoint(
        {item.generation_seed for item in screening.entries}
    )
    assert {item.image_lineage_digest for item in fit.entries}.isdisjoint(
        {item.image_lineage_digest for item in screening.entries}
    )


@pytest.mark.unit
def test_lf_whitened_manifests_do_not_reuse_registered_experiment_axes() -> None:
    _protocol, fit, screening = load_lf_whitened_score_screening_protocol(
        PROTOCOL, repository_root=ROOT
    )
    current_paths = {
        Path("configs/experiments/lf_whitening_null_fit_manifest.json"),
        Path("configs/experiments/lf_whitened_score_screening_manifest.json"),
    }
    existing_prompts: set[str] = set()
    existing_prompt_digests: set[str] = set()
    existing_seeds: set[int] = set()
    existing_lineages: set[str] = set()
    for path in (ROOT / "configs/experiments").glob("*manifest*.json"):
        if path.relative_to(ROOT) in current_paths:
            continue
        value = json.loads(path.read_text("utf-8"))
        stack = [value]
        while stack:
            item = stack.pop()
            if type(item) is dict:
                prompt = item.get("prompt", item.get("prompt_text"))
                if type(prompt) is str:
                    existing_prompts.add(prompt)
                if type(item.get("prompt_digest")) is str:
                    existing_prompt_digests.add(item["prompt_digest"])
                if type(item.get("generation_seed")) is int:
                    existing_seeds.add(item["generation_seed"])
                if type(item.get("image_lineage_digest")) is str:
                    existing_lineages.add(item["image_lineage_digest"])
                stack.extend(item.values())
            elif type(item) is list:
                stack.extend(item)
    entries = (*fit.entries, *screening.entries)
    assert {item.prompt for item in entries}.isdisjoint(existing_prompts)
    assert {item.prompt_digest for item in entries}.isdisjoint(
        existing_prompt_digests
    )
    assert {item.generation_seed for item in entries}.isdisjoint(existing_seeds)
    assert {item.image_lineage_digest for item in entries}.isdisjoint(
        existing_lineages
    )


@pytest.mark.unit
def test_lf_clean_null_fit_freezes_one_binary32_whitening_asset() -> None:
    rows = tuple(
        tuple(float((cluster + 1) * (index + 1)) for index in range(96))
        for cluster in range(32)
    )
    asset = fit_lf_null_whitening_asset(
        rows, fit_manifest_sha256="a" * 64
    )
    replay = fit_lf_null_whitening_asset(
        rows, fit_manifest_sha256="a" * 64
    )

    assert len(asset.weights_binary32_be_hex) == 96
    assert asset.whitening_asset_digest == replay.whitening_asset_digest
    assert asset.canonical_payload == replay.canonical_payload
    assert asset.fit_manifest_sha256 == "a" * 64
    with pytest.raises(LfWhitenedScoreMetricError, match="coverage"):
        fit_lf_null_whitening_asset(
            rows[:-1], fit_manifest_sha256="a" * 64
        )


@pytest.mark.unit
def test_lf_clean_null_sufficient_statistics_execute_frozen_dct_rings() -> None:
    values: list[float] = []
    for channel in range(16):
        for height in range(64):
            for width in range(64):
                values.append(
                    float(
                        (channel + 1) * 0.001
                        + cos(pi * (height + 0.5) * 4.0 / 64.0)
                        * (1.0 if channel == 0 else 0.01)
                        + width * width * 1e-7
                    )
                )
    energy = clean_null_band_energy_sums(tuple(values))

    assert len(energy) == 96
    assert all(value >= 0.0 for value in energy)
    assert energy[2] > energy[0]
    assert sum(energy) > 0.0
    with pytest.raises(
        LfWhitenedScoreMetricError, match="global clean null variance"
    ):
        fit_lf_null_whitening_asset(
            ((0.0,) * 96,) * 32, fit_manifest_sha256="a" * 64
        )


@pytest.mark.unit
def test_lf_whitened_screening_uses_max_of_four_wrong_as_one_cluster() -> None:
    observations = tuple(
        create_lf_whitened_screening_observation(
            cluster_ordinal=index,
            raw_registered_score=0.1,
            raw_primary_null_score=0.0,
            raw_wrong_key_scores=(0.09, 0.08, 0.07, 0.06),
            whitened_registered_score=0.3,
            whitened_primary_null_score=0.0,
            whitened_wrong_key_scores=(0.1, 0.09, 0.08, 0.07),
            whitening_asset_digest="a" * 64,
            raw_detector_config_digest="b" * 64,
            whitened_detector_config_digest="c" * 64,
        )
        for index in range(8)
    )
    decision = evaluate_lf_whitened_screening(
        observations, integrity_failure_count=0, margin_floor=MARGIN_FLOOR
    )

    assert decision.cluster_count == 8
    assert decision.registered_primary_null_pass_count == 8
    assert decision.registered_max_wrong_pass_count == 8
    assert decision.positive_raw_to_whitened_improvement_count == 8
    assert decision.allow_request_for_lf_whitened_directional_validation is True
    assert len(observations) == 8
    assert all(len(item.whitened_wrong_key_scores) == 4 for item in observations)


@pytest.mark.unit
def test_lf_clean_runtime_records_create_once_asset_from_verified_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, fit, screening = load_lf_whitened_score_screening_protocol(
        PROTOCOL, repository_root=ROOT
    )
    callbacks = tuple(tuple(range(20)) for _ in range(33))
    backend = FakeContentBackend(callback_sequences=callbacks)  # type: ignore[arg-type]
    runtime = create_runtime_adapter(backend)
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        screening_manifest_digest=screening.digest(),
        key_family_namespace=screening.key_family_namespace,
    )
    root_public = identify_root_key(registered_root).root_key_public_digest
    candidate_config_digest = canonical_digest(
        {
            "adapter": adapter.configuration.config_digest,
            "fit": fit.digest(),
            "runtime": runtime.session.runtime_config_digest,
            "screening": screening.digest(),
        }
    )
    authority = canonical_digest(
        {"protocol": protocol.digest(), "run_id": RUN_ID}
    )
    runner = LfWhitenedScoreScreeningRunner(
        protocol=protocol,
        null_fit_manifest=fit,
        screening_manifest=screening,
        adapter=adapter,
        runtime_adapter=runtime,
        method_code_revision="a" * 40,
        run_id=RUN_ID,
        registered_root_key=registered_root,
        root_key_public_digest=root_public,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest=authority,
        candidate_config_digest=candidate_config_digest,
    )
    identity = FrozenWorkerIdentity(
        revision="a" * 40,
        protocol_digest=protocol.digest(),
        execution_intent_authority_digest=authority,
        input_manifest_digest=canonical_digest(
            {"fit": fit.digest(), "screening": screening.digest()}
        ),
        candidate_config_digest=candidate_config_digest,
        unit_roster_digest=protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=RUN_ID,
        worker_identity=identity,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    monkeypatch.setattr(
        screening_runner_module,
        "clean_null_band_energy_sums",
        lambda _values: tuple(float(index + 1) for index in range(96)),
    )
    lease = store.acquire_lease(
        session_id="lf_whitening_fit_session",
        now_epoch_seconds=100,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    base = torch.linspace(
        -1.0, 1.0, steps=16 * 64 * 64, dtype=torch.float32
    ).reshape(1, 16, 64, 64).to(torch.float16)
    operational_intent = store.create_session_intent(
        cursor, lease, now_epoch_seconds=101
    )
    operational_record = runner.execute_operational_smoke(
        base_latent=base,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    assert type(operational_record) is DevelopmentOperationalRecord
    assert operational_record.counts_as_scientific_coverage is False
    store.commit_session_unit(
        cursor,
        lease,
        operational_intent,
        record=operational_record,
        raw_secret_values=(ROOT_KEY, registered_root),
        now_epoch_seconds=102,
    )
    for index in range(32):
        intent = store.create_session_intent(
            cursor, lease, now_epoch_seconds=103 + index * 2
        )
        evidence = store.verified_terminal_scientific_evidence(
            now_epoch_seconds=103 + index * 2
        )
        record = runner.execute_null_fit_cluster(
            cluster_ordinal=index,
            base_latent=base,
            attempt_index=0,
            retry_parent_intent_digest=None,
            maximum_duration_seconds=2700,
            prior_verified_fit_evidence=evidence,
        )
        store.commit_session_unit(
            cursor,
            lease,
            intent,
            record=record,
            raw_secret_values=(ROOT_KEY, registered_root),
            now_epoch_seconds=104 + index * 2,
        )
    evidence = store.verified_terminal_scientific_evidence(
        now_epoch_seconds=200
    )
    asset = runner.replay_whitening_asset(evidence)

    assert len(evidence) == 32
    assert asset.fit_manifest_sha256 == protocol.null_fit_manifest_file_sha256
    assert len(asset.weights_binary32_be_hex) == 96
    assert evidence[-1][0].operation_result_payload[
        "whitening_asset_digest"
    ] == asset.whitening_asset_digest
    assert all(
        record.operation_result_payload["whitening_asset_payload"] is None
        for record, _marker in evidence[:-1]
    )
    assert runtime.session.selected_device == "cpu"
    assert backend.run_calls == 33
    assert all(
        record.operation_result_payload["clean_observation_protocol"]
        == "final_image_vae_posterior_mode"
        for record, _marker in evidence
    )
    runtime.close()


@pytest.mark.unit
def test_lf_whitening_entrypoint_derives_separate_screening_key_family() -> None:
    protocol, _fit, screening = load_lf_whitened_score_screening_protocol(
        PROTOCOL, repository_root=ROOT
    )
    derived = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        screening_manifest_digest=screening.digest(),
        key_family_namespace=screening.key_family_namespace,
    )

    assert identify_root_key(derived).root_key_public_digest != identify_root_key(
        ROOT_KEY
    ).root_key_public_digest
    assert derived.startswith("ceg-wm-lf-whitened-screening-registered:")


@pytest.mark.unit
def test_lf_whitening_production_entrypoint_retries_then_commits_fit_and_screening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends = [
        _EntrypointBackend(
            callback_count=2,
            failure_run_index=1,
            failure_factory=lambda: MemoryError("bounded test resource failure"),
        ),
        _EntrypointBackend(
            callback_count=48,
            failure_run_index=46,
            failure_factory=lambda: ValueError(
                "bounded test terminal screening failure"
            ),
        ),
    ]
    monkeypatch.setattr(
        lf_whitened_score_screening_entrypoint,
        "Sd35PipelineBackend",
        lambda **_kwargs: backends.pop(0),
    )
    initialize = Sd35RuntimeAdapter.initialize
    monkeypatch.setattr(
        Sd35RuntimeAdapter,
        "initialize",
        lambda self, _device: initialize(self, "cpu"),
    )
    base = torch.ones((1, 16, 64, 64), dtype=torch.float16)
    monkeypatch.setattr(
        lf_whitened_score_screening_entrypoint,
        "_base_latent",
        lambda _seed, **_kwargs: base.detach().clone(),
    )
    monkeypatch.setattr(
        screening_runner_module,
        "clean_null_band_energy_sums",
        lambda _values: tuple(float(index + 1) for index in range(96)),
    )

    raw_call_count = {"value": 0}
    whitened_call_count = {"value": 0}

    def fake_raw(self, observation, detection_key):
        position = raw_call_count["value"] % 6
        raw_call_count["value"] += 1
        wrong_index = (
            detection_key.wrong_key_index
            if type(detection_key) is DerivedWrongKeyMaterial
            else None
        )
        score = 0.2 if position == 0 else (0.0 if position == 1 else 0.1)
        result = LfDetectionResult(
            candidate_id="lf_low_pass",
            candidate_ids=("key_schedule_sha256_counter", "lf_low_pass"),
            lf_score=score,
            detector_identity="a" * 64,
            detector_config_digest="b" * 64,
            root_key_public_digest="c" * 64,
            key_role="wrong" if wrong_index is not None else "registered",
            wrong_key_index=wrong_index,
            observation_digest=observation.observation_digest,
            template_digest=canonical_digest(
                {"raw_template_wrong_key_index": wrong_index}
            ),
        )
        return ComponentCallObservation(
            responsibility="lf_detector",
            public_callable="main.lf_detector",
            adapter_config_digest=self.configuration.config_digest,
            result_type="LfDetectionResult",
            result_identity=result.detector_identity,
            upstream_runtime_identity=None,
            result=result,
        )

    def fake_whitened(self, observation, detection_key, whitening_asset):
        position = whitened_call_count["value"] % 6
        whitened_call_count["value"] += 1
        wrong_index = (
            detection_key.wrong_key_index
            if type(detection_key) is DerivedWrongKeyMaterial
            else None
        )
        score = 0.4 if position == 0 else (0.0 if position == 1 else 0.1)
        result = LfNullWhitenedDetectionResult(
            candidate_id="lf_null_whitened_matched_score",
            candidate_ids=(
                "key_schedule_sha256_counter",
                "lf_low_pass",
                "lf_null_whitened_matched_score",
            ),
            lf_score=score,
            detector_identity="d" * 64,
            detector_config_digest="e" * 64,
            whitening_asset_digest=whitening_asset.whitening_asset_digest,
            root_key_public_digest="c" * 64,
            key_role="wrong" if wrong_index is not None else "registered",
            wrong_key_index=wrong_index,
            observation_digest=observation.observation_digest,
            template_digest=canonical_digest(
                {"whitened_template_wrong_key_index": wrong_index}
            ),
        )
        return ComponentCallObservation(
            responsibility="lf_detector",
            public_callable="main.lf_null_whitened_matched_detector",
            adapter_config_digest=self.configuration.config_digest,
            result_type="LfNullWhitenedDetectionResult",
            result_identity=result.detector_identity,
            upstream_runtime_identity=None,
            result=result,
        )

    monkeypatch.setattr(CegWmExperimentAdapter, "detect_lf", fake_raw)
    monkeypatch.setattr(
        CegWmExperimentAdapter,
        "detect_lf_null_whitened",
        fake_whitened,
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda _index: 1
    )
    environment = {
        "CEG_WM_ROOT_KEY": ROOT_KEY,
        "HF_TOKEN": "lf_whitening_test_token",
    }

    first_code, first = (
        lf_whitened_score_screening_entrypoint.execute_lf_whitened_score_screening_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_whitening_retry_session",
            execution_package_sha256="f" * 64,
            environment=environment,
        )
    )
    second_code, second = (
        lf_whitened_score_screening_entrypoint.execute_lf_whitened_score_screening_session(
            repository_root=ROOT,
            expected_revision="a" * 40,
            persistent_root=tmp_path / "persistent",
            cache_root=tmp_path / "cache",
            run_id=RUN_ID,
            session_id="lf_whitening_terminal_session",
            execution_package_sha256="f" * 64,
            environment=environment,
        )
    )

    assert first_code == 0
    assert first["termination_reason"] == "retryable_resource_failure"
    assert first["committed_unit_count"] == 2
    assert second_code == 0
    assert second["termination_reason"] == "frozen_roster_complete"
    assert second["committed_unit_count"] == 42
    assert second["screening_decision"].get("decision_available") is not False, (
        second["screening_decision"].get("failure_reason")
    )
    assert second["screening_decision"]["integrity_failure_count"] == 1
    assert (
        second["screening_decision"][
            "allow_request_for_lf_whitened_directional_validation"
        ]
        is False
    )
    run_root = tmp_path / "persistent" / RUN_ID
    assert len(tuple((run_root / "markers").glob("*.COMMITTED.json"))) == 42
    bundles = tuple((run_root / "bundles").glob("sha256_*.zip"))
    assert len(bundles) == 42
    scientific_records: dict[int, list[dict[str, object]]] = {}
    for bundle in bundles:
        with ZipFile(bundle) as source:
            if "records/development_scientific_record.json" not in source.namelist():
                continue
            record = json.loads(
                source.read("records/development_scientific_record.json")
            )
            scientific_records.setdefault(record["unit_index"], []).append(record)
    assert tuple(sorted(scientific_records)) == tuple(range(1, 41))
    unit_one = sorted(
        scientific_records[1], key=lambda record: record["attempt_index"]
    )
    assert [record["execution_status"] for record in unit_one] == [
        "retry",
        "success",
    ]
    assert unit_one[1]["retry_parent_intent_digest"]
    assert all(
        scientific_records[index][0]["execution_status"] == "success"
        for index in range(33, 40)
    )
    assert scientific_records[40][0]["execution_status"] == "failed"
