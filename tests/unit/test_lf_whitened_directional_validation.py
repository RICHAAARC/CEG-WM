"""Frozen identity checks for LF whitened directional validation."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path

import pytest
import torch

from experiments.methods import (
    CegWmExperimentAdapter,
    load_ceg_wm_experiment_adapter_configuration,
)
from experiments.metrics.binomial import clopper_pearson_lower
from experiments.metrics.lf_whitened_directional_validation import (
    aggregate_lf_whitened_direction,
    create_lf_whitened_directional_observation,
)
from experiments.metrics.lf_whitened_score_screening import (
    fit_lf_null_whitening_asset,
)
from experiments.protocol.development_records import (
    DEVELOPMENT_CLAIM_BOUNDARY,
    OPERATIONAL_RECORD_COLLECTION_ROLE,
    OPERATIONAL_RECORD_KIND,
    OPERATIONAL_RECORD_SCHEMA,
    DevelopmentOperationalRecord,
    DevelopmentScientificRecord,
    canonical_development_value_digest,
)

from experiments.protocol.hf_only_detector_directional_validation import (
    load_authority_deny_axes,
)
from experiments.protocol.lf_whitened_directional_validation import (
    FUTURE_SPLIT_EXCLUSION_ROLES,
    LF_DIRECTIONAL_COMPONENT_IDS,
    PRACTICAL_MARGIN_FLOOR,
    LfWhitenedDirectionalProtocolError,
    canonical_digest,
    derive_lf_whitened_directional_analysis_identity,
    load_lf_whitened_directional_validation_protocol,
)
from experiments.protocol.lf_whitened_score_screening import (
    load_lf_whitened_score_screening_protocol,
)
from experiments.runners.development_persistence import (
    CommittedUnit,
    DevelopmentPersistentStore,
    FrozenWorkerIdentity,
)
from experiments.runners.lf_whitened_directional_validation import (
    LfWhitenedDirectionalRunnerError,
    LfWhitenedDirectionalValidationRunner,
    _observation,
)
from main import (
    LfDetectionObservation,
    LfNullWhiteningAsset,
    derive_wrong_key_material,
    identify_root_key,
    lf_carrier,
)
from runtime import create_runtime_adapter
from scripts.experiment_execution.lf_whitened_directional_validation_entrypoint import (
    _derive_registered_experiment_root,
)
from scripts.experiment_execution.component_source_closure import (
    build_component_source_closure,
)
from tests.unit.test_runtime_content_write_and_vae import FakeContentBackend
from tests.helpers.historical_repository import materialize_historical_repository


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/lf_whitened_directional_validation.json"
LF_DIRECTIONAL_PRODUCER_REVISION = "51adb765cdddafcb4c65c357e899c77b4c9f36d2"
LF_DIRECTIONAL_PRODUCER_PATHS = (
    ".codex/research_state/method_readiness.yaml",
    "configs/experiments/lf_whitened_directional_validation.json",
    "docs/design/candidate_specifications.md",
    "main/shared/key_schedule.py",
    "main/content_chain/lf_carrier.py",
    "main/content_chain/embedder.py",
    "main/content_chain/lf_detector.py",
    "main/content_chain/lf_whitening.py",
)
SCREENING_CONFIG = ROOT / "configs/experiments/lf_whitened_score_screening.json"
COMPONENTS = ROOT / "configs/experiments/internal_execution_components.json"
ROOT_KEY = "ceg-wm-lf-whitened-directional-test-key"


def _load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text("utf-8"))


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_lf_whitened_directional_protocol_freezes_budget_controls_and_gate() -> None:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )

    assert protocol.operational_unit_count == 1
    assert protocol.run_id == (
        "ceg_wm_lf_whitened_directional_validation_prepared_feature_execution"
    )
    assert protocol.scientific_cluster_count == 32
    assert protocol.maximum_total_units == 33
    assert tuple(unit.unit_index for unit in protocol.unit_roster) == tuple(
        range(33)
    )
    assert protocol.unit_roster[0].responsibility_id == (
        "lf_whitened_detector_runtime_preflight"
    )
    assert {
        unit.responsibility_id for unit in protocol.unit_roster[1:]
    } == {"lf_detector"}
    assert protocol.wrong_key_roster_size == 4
    assert protocol.practical_margin_floor == PRACTICAL_MARGIN_FLOOR
    assert (
        protocol.content_relative_l2_numerator,
        protocol.content_relative_l2_denominator,
    ) == (3, 250)
    assert protocol.minimum_registered_minus_null_success_count == 28
    assert protocol.minimum_registered_minus_max_wrong_success_count == 28
    assert protocol.confidence_level == 0.95
    assert protocol.confidence_lower_bound_requirement == (
        "strictly_greater_than_one_half"
    )
    assert protocol.passing_module_outcome == "mechanism_signal_observed"
    assert protocol.passing_candidate_recommendation == (
        "candidate_worth_further_selection"
    )
    assert "no_threshold" in protocol.claim_boundary
    assert "no_fpr" in protocol.claim_boundary
    assert "no_promotion" in protocol.claim_boundary
    assert len(manifest.entries) == 32
    assert {item.split for item in manifest.entries} == {"development"}
    assert {item.role_id for item in manifest.entries} == {
        "lf_whitened_directional_validation"
    }
    assert protocol.future_split_exclusion_roles == FUTURE_SPLIT_EXCLUSION_ROLES


@pytest.mark.unit
def test_lf_whitened_directional_manifest_is_disjoint_from_prior_authorities() -> None:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    prior = load_authority_deny_axes(
        protocol.prior_development_manifests, ROOT
    )

    assert {item.prompt_digest for item in manifest.entries}.isdisjoint(
        prior.prompt_digests
    )
    assert {
        manifest.source_cluster_namespace,
        *(item.cluster_identity for item in manifest.entries),
    }.isdisjoint(prior.source_cluster_identities)
    assert {manifest.seed_namespace}.isdisjoint(prior.seed_namespaces)
    assert {item.generation_seed for item in manifest.entries}.isdisjoint(
        prior.generation_seeds
    )
    assert {
        manifest.image_lineage_namespace,
        *(item.image_lineage_identity for item in manifest.entries),
        *(item.image_lineage_digest for item in manifest.entries),
    }.isdisjoint(prior.image_lineage_identities)
    assert {
        manifest.key_family_namespace,
        protocol.registered_key_derivation_identity,
        protocol.wrong_key_control_identity,
    }.isdisjoint(prior.key_control_identities)


@pytest.mark.unit
def test_lf_whitened_directional_component_authority_replays_reviewed_sources(
    tmp_path: Path,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("detached research copy lacks local Git producer objects")
    producer_root = materialize_historical_repository(
        source_root=ROOT,
        revision=LF_DIRECTIONAL_PRODUCER_REVISION,
        destination=tmp_path / "lf-directional-producer",
        paths=LF_DIRECTIONAL_PRODUCER_PATHS,
    )
    producer_authority = json.loads(
        (
            producer_root
            / "configs/experiments/lf_whitened_directional_validation.json"
        ).read_text(encoding="utf-8")
    )
    producer_readiness = json.loads(
        (
            producer_root / ".codex/research_state/method_readiness.yaml"
        ).read_text(encoding="utf-8")
    )
    producer_closure = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS,
        producer_readiness["components"],
        producer_root,
    )
    assert tuple(producer_authority["ordered_component_ids"]) == (
        producer_closure.ordered_component_ids
    )
    assert tuple(producer_authority["component_source_bindings"]) == tuple(
        asdict(item) for item in producer_closure.source_bindings
    )
    assert producer_authority["component_implementation_digest"] == (
        producer_closure.component_implementation_digest
    )
    assert producer_authority["candidate_specification_sha256"] == sha256(
        (producer_root / "docs/design/candidate_specifications.md").read_bytes()
    ).hexdigest()
    frozen_review_reference = (
        "independent_lf_prepared_feature_semantic_review:"
        "019fe0f3-b8e8-7230-98f1-9ae0450c1f4a:"
        "00bed2baaf60f039868c208291c86b539a54b2f3:APPROVE"
    )
    frozen_reviewed_revision = "00bed2baaf60f039868c208291c86b539a54b2f3"
    producer_review = producer_readiness["independent_semantic_review"]
    assert producer_review["decision"] == "approve"
    assert producer_review["review_reference"] == producer_authority[
        "method_review_reference"
    ]
    assert producer_review["reviewed_repository_revision"] == producer_authority[
        "method_reviewed_revision"
    ]
    assert producer_review["candidate_specification_sha256"] == producer_authority[
        "candidate_specification_sha256"
    ]
    assert producer_authority["method_review_reference"] == frozen_review_reference
    assert producer_authority["method_reviewed_revision"] == frozen_reviewed_revision
@pytest.mark.unit
def test_lf_directional_producer_replay_rejects_candidate_specification_tampering(
    tmp_path: Path,
) -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("detached research copy lacks local Git producer objects")
    producer_root = materialize_historical_repository(
        source_root=ROOT,
        revision=LF_DIRECTIONAL_PRODUCER_REVISION,
        destination=tmp_path / "lf-directional-producer",
        paths=LF_DIRECTIONAL_PRODUCER_PATHS,
    )
    authority = json.loads(
        (
            producer_root
            / "configs/experiments/lf_whitened_directional_validation.json"
        ).read_text(encoding="utf-8")
    )
    candidate_specification = producer_root / "docs/design/candidate_specifications.md"
    candidate_specification.write_bytes(
        candidate_specification.read_bytes() + b"\nproducer tamper\n"
    )

    assert sha256(candidate_specification.read_bytes()).hexdigest() != authority[
        "candidate_specification_sha256"
    ]


@pytest.mark.unit
def test_lf_whitening_asset_fit_authority_remains_bound_to_completed_producer() -> None:
    protocol, _manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    screening, _fit, _screen = load_lf_whitened_score_screening_protocol(
        SCREENING_CONFIG,
        repository_root=ROOT,
    )

    assert protocol.whitening_asset_fit_producer_revision == (
        "a78c47184cf83ad351bb4442ebd31c218726de25"
    )
    assert protocol.whitening_asset_fit_identity == screening.protocol_id
    assert protocol.whitening_asset_fit_run_id == screening.run_id
    assert protocol.whitening_asset_fit_protocol_digest == screening.digest()
    assert protocol.whitening_null_fit_manifest_file_sha256 == sha256(
        (ROOT / screening.null_fit_manifest_path).read_bytes()
    ).hexdigest()


@pytest.mark.unit
def test_lf_whitened_directional_analysis_identities_cover_unique_clusters() -> None:
    _protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    identities = tuple(
        derive_lf_whitened_directional_analysis_identity(
            entry,
            manifest,
            key_family_digest="f" * 64,
        )
        for entry in manifest.entries
    )

    assert len({item.unit_id for item in identities}) == 32
    assert len({item.source_cluster_id for item in identities}) == 32
    assert {item.generation_seed for item in identities} == {
        item.generation_seed for item in manifest.entries
    }


@pytest.mark.unit
def test_lf_whitened_directional_protocol_rejects_gate_or_component_drift(
    tmp_path: Path,
) -> None:
    payload = _load_config()
    payload["minimum_registered_minus_null_success_count"] = 27
    gate_path = tmp_path / "gate.json"
    _write_config(gate_path, payload)
    with pytest.raises(
        LfWhitenedDirectionalProtocolError, match="scientific gate"
    ):
        load_lf_whitened_directional_validation_protocol(
            gate_path, repository_root=ROOT
        )

    payload = _load_config()
    bindings = payload["component_source_bindings"]
    assert type(bindings) is list
    detector_binding = bindings[3]
    assert type(detector_binding) is dict
    detector_binding["source_sha256"] = "0" * 64
    closure_path = tmp_path / "closure.json"
    _write_config(closure_path, payload)
    with pytest.raises(
        LfWhitenedDirectionalProtocolError, match="implementation digest"
    ):
        load_lf_whitened_directional_validation_protocol(
            closure_path, repository_root=ROOT
        )


@pytest.mark.unit
def test_lf_whitened_directional_protocol_rejects_prior_prompt_reuse(
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (ROOT / "configs/experiments/lf_whitened_directional_validation_manifest.json").read_text(
            "utf-8"
        )
    )
    prior = json.loads(
        (ROOT / "configs/experiments/lf_transmission_diagnostic_manifest.json").read_text(
            "utf-8"
        )
    )
    prior_prompt = prior["entries"][0]["prompt"]
    manifest["entries"][0]["prompt"] = prior_prompt
    manifest["entries"][0]["prompt_digest"] = sha256(
        prior_prompt.encode("utf-8")
    ).hexdigest()
    manifest_path = tmp_path / "overlap_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    payload = _load_config()
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_file_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    config_path = tmp_path / "overlap_config.json"
    _write_config(config_path, payload)

    with pytest.raises(LfWhitenedDirectionalProtocolError, match="overlaps"):
        load_lf_whitened_directional_validation_protocol(
            config_path, repository_root=ROOT
        )


def _asset() -> LfNullWhiteningAsset:
    fit = fit_lf_null_whitening_asset(
        tuple(
            tuple(float((cluster + 1) * (index + 1)) for index in range(96))
            for cluster in range(32)
        ),
        fit_manifest_sha256="a" * 64,
    )
    return LfNullWhiteningAsset.from_canonical_payload(
        fit.canonical_payload,
        whitening_asset_digest=fit.whitening_asset_digest,
    )


def _lf_base_latent() -> torch.Tensor:
    return torch.linspace(
        -1.0,
        1.0,
        steps=16 * 64 * 64,
        dtype=torch.float32,
    ).reshape(1, 16, 64, 64).to(torch.float16)


def _directional_runner() -> tuple[
    LfWhitenedDirectionalValidationRunner,
    object,
]:
    protocol, manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    runtime = create_runtime_adapter(
        FakeContentBackend(
            callback_sequences=tuple(tuple(range(20)) for _ in range(8))
        )  # type: ignore[arg-type]
    )
    runtime.initialize("cpu")
    adapter = CegWmExperimentAdapter(
        load_ceg_wm_experiment_adapter_configuration(COMPONENTS)
    )
    registered_root = _derive_registered_experiment_root(
        ROOT_KEY,
        protocol_digest=protocol.digest(),
        manifest_digest=manifest.digest(),
        key_family_namespace=manifest.key_family_namespace,
    )
    public_root = identify_root_key(registered_root).root_key_public_digest
    return (
        LfWhitenedDirectionalValidationRunner(
            protocol=protocol,
            manifest=manifest,
            adapter=adapter,
            runtime_adapter=runtime,
            whitening_asset=_asset(),
            method_code_revision="a" * 40,
            run_id=protocol.run_id,
            registered_root_key=registered_root,
            root_key_public_digest=public_root,
            protocol_digest=protocol.digest(),
            execution_intent_authority_digest="b" * 64,
            candidate_config_digest="c" * 64,
        ),
        runtime,
    )


def _validated_operational_record(
    runner: LfWhitenedDirectionalValidationRunner,
) -> DevelopmentOperationalRecord:
    elapsed_seconds = 0.01
    result_payload = {
        "operational_role": "environment_runtime_throughput_preflight",
        "source_cluster_ordinal": 0,
        "case_ids": ["formal_record_persistence_contract"],
        "responsibility_result_digests": [
            ["content_embedder", canonical_digest({"formal_record": "operational"})]
        ],
        "elapsed_seconds": elapsed_seconds,
        "runtime_config_digest": canonical_digest(
            {"runtime": "lf_directional_persistence_fixture"}
        ),
        "counts_as_scientific_coverage": False,
        "scientific_claims_supported": False,
    }
    provisional = DevelopmentOperationalRecord(
        schema_version=OPERATIONAL_RECORD_SCHEMA,
        collection_role=OPERATIONAL_RECORD_COLLECTION_ROLE,
        record_kind=OPERATIONAL_RECORD_KIND,
        record_id="0" * 64,
        run_id=runner.run_id,
        protocol_digest=runner.protocol_digest,
        method_code_revision=runner.method_code_revision,
        unit_index=0,
        phase="development_environment_preflight",
        source_cluster_ordinal=0,
        candidate_config_digest=runner.candidate_config_digest,
        attempt_index=0,
        retry_parent_intent_digest=None,
        actual_elapsed_seconds=elapsed_seconds,
        maximum_duration_seconds=2700,
        operation_result_payload=result_payload,
        counts_as_scientific_coverage=False,
        scientific_claims_supported=False,
        scientific_claim_boundary=DEVELOPMENT_CLAIM_BOUNDARY,
    )
    record = replace(
        provisional,
        record_id=canonical_development_value_digest(
            provisional.payload_without_record_id()
        ),
    )
    record.validate()
    return record


@pytest.mark.unit
def test_lf_whitened_directional_runner_uses_public_detector_and_four_wrong_controls() -> None:
    runner, runtime = _directional_runner()
    scientific = runner.execute_scientific_cluster(
        cluster_ordinal=0,
        base_latent=_lf_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    runtime.close()

    result = scientific.operation_result_payload["directional_observation"]
    assert len(result["wrong_key_scores"]) == 4
    assert scientific.detector_trace["public_callable"] == (
        "main.lf_null_whitened_matched_detector"
    )
    assert scientific.detector_trace["same_image_registered_four_wrong_reuse"] is True
    assert scientific.detector_trace["paired_clean_primary_null"] is True
    assert scientific.detector_trace["reference_image_used"] is False
    assert scientific.detector_trace["embed_record_used"] is False
    assert scientific.detector_trace["private_latent_used_by_detector"] is False
    assert scientific.threshold_trace["raw_threshold_identity"] is None
    assert scientific.module_outcome is None
    assert scientific.candidate_recommendation is None


@pytest.mark.unit
def test_lf_whitened_directional_prepared_record_payload_matches_legacy_calls() -> None:
    runner, runtime = _directional_runner()
    runtime_result = runner._execute_paired_runtime(_lf_base_latent())
    _measurement, prepared_payload = runner._detect_public_pair(
        runtime_result,
        cluster_ordinal=0,
    )
    candidate = _observation(runtime_result.watermarked_detection_latent)
    clean = _observation(runtime_result.clean_detection_latent)
    registered = runner.adapter.detect_lf_null_whitened(
        candidate,
        runner.registered_root_key,
        runner.whitening_asset,
    ).result
    primary_null = runner.adapter.detect_lf_null_whitened(
        clean,
        runner.registered_root_key,
        runner.whitening_asset,
    ).result
    wrong = tuple(
        runner.adapter.detect_lf_null_whitened(
            candidate,
            derive_wrong_key_material(runner.root_key_public_digest, index),
            runner.whitening_asset,
        ).result
        for index in range(runner.protocol.wrong_key_roster_size)
    )
    runtime.close()

    legacy_payload = {
        "registered": asdict(registered),
        "primary_null": asdict(primary_null),
        "wrong_keys": tuple(asdict(item) for item in wrong),
    }
    assert prepared_payload == legacy_payload
    assert canonical_digest(prepared_payload) == canonical_digest(
        legacy_payload
    )
    assert "prepared" not in json.dumps(
        prepared_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


@pytest.mark.unit
def test_lf_whitened_directional_runner_reuses_deterministic_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_runner, baseline_runtime = _directional_runner()
    runtime_result = baseline_runner._execute_paired_runtime(_lf_base_latent())
    candidate = LfDetectionObservation.from_public_image_encoding(
        tuple(
            float(item)
            for item in runtime_result.watermarked_detection_latent.detach()
            .cpu()
            .float()
            .reshape(-1)
        ),
        tuple(int(size) for size in runtime_result.watermarked_detection_latent.shape),
    )
    clean = LfDetectionObservation.from_public_image_encoding(
        tuple(
            float(item)
            for item in runtime_result.clean_detection_latent.detach()
            .cpu()
            .float()
            .reshape(-1)
        ),
        tuple(int(size) for size in runtime_result.clean_detection_latent.shape),
    )
    lf_detector_module = import_module("main.content_chain.lf_detector")
    method_module = import_module("experiments.methods.ceg_wm")
    original_dct = lf_detector_module._affine_detrended_dct
    original_public_detector = method_module.lf_null_whitened_matched_detector
    coefficient_by_input_digest = {
        lf_detector_module._digest(candidate.values): original_dct(
            candidate.values,
            role="candidate count fixture",
        ),
        lf_detector_module._digest(clean.values): original_dct(
            clean.values,
            role="clean count fixture",
        ),
    }
    registered_carrier = lf_carrier(
        baseline_runner.registered_root_key,
        (1, 16, 64, 64),
    )
    carrier_by_wrong_index = {None: registered_carrier}
    coefficient_by_input_digest[
        lf_detector_module._digest(registered_carrier.template)
    ] = baseline_runner._registered_prepared_template.coefficients
    for index, prepared in enumerate(
        baseline_runner._wrong_prepared_templates
    ):
        wrong_carrier = lf_carrier(
            derive_wrong_key_material(
                baseline_runner.root_key_public_digest,
                index,
            ),
            (1, 16, 64, 64),
        )
        coefficient_by_input_digest[
            lf_detector_module._digest(wrong_carrier.template)
        ] = prepared.coefficients
        carrier_by_wrong_index[index] = wrong_carrier
    counts = {"dct": 0, "public_detector": 0}

    def counted_dct(values, *, role):
        counts["dct"] += 1
        coefficients = coefficient_by_input_digest[
            lf_detector_module._digest(values)
        ]
        return coefficients.copy(order="C")

    def counted_public_detector(*args, **kwargs):
        counts["public_detector"] += 1
        return original_public_detector(*args, **kwargs)

    monkeypatch.setattr(
        lf_detector_module,
        "_affine_detrended_dct",
        counted_dct,
    )
    monkeypatch.setattr(
        method_module,
        "lf_null_whitened_matched_detector",
        counted_public_detector,
    )
    monkeypatch.setattr(
        lf_detector_module,
        "_whitened_cosine",
        lambda observation_coefficients, template_coefficients, asset: 0.25,
    )
    monkeypatch.setattr(
        lf_detector_module,
        "lf_carrier",
        lambda detection_key, shape, mask_lf, model_revision: (
            carrier_by_wrong_index[
                None
                if isinstance(detection_key, str)
                else detection_key.wrong_key_index
            ]
        ),
    )
    runner, runtime = _directional_runner()
    for cluster_ordinal in range(32):
        runner._detect_public_pair(
            runtime_result,
            cluster_ordinal=cluster_ordinal,
        )
    runtime.close()
    baseline_runtime.close()

    assert counts == {"dct": 69, "public_detector": 192}
    assert 69 == 2 * 32 + 1 + 4
    assert 384 == 32 * (2 + 1 + 1 + 4 + 4)


def _metric_observation(index: int, *, passed: bool):
    registered = 0.4 if passed else 0.0
    return create_lf_whitened_directional_observation(
        cluster_ordinal=index,
        registered_score=registered,
        primary_null_score=0.0,
        wrong_key_scores=(0.1, 0.09, 0.08, 0.07),
        candidate_observation_digest=canonical_digest({"candidate": index}),
        clean_observation_digest=canonical_digest({"clean": index}),
        registered_detector_identity="d" * 64,
        primary_null_detector_identity="d" * 64,
        wrong_key_detector_identities=("d" * 64,) * 4,
        detector_config_digest="e" * 64,
        observation_protocol="final_image_vae_posterior_mode",
        whitening_asset_digest="f" * 64,
        registered_template_digest=canonical_digest({"registered": index}),
        primary_null_template_digest=canonical_digest({"registered": index}),
        wrong_key_template_digests=tuple(
            canonical_digest({"wrong": wrong, "cluster": index})
            for wrong in range(4)
        ),
        registered_root_key_public_digest="1" * 64,
        wrong_key_indexes=(0, 1, 2, 3),
        materialization_integrity_status="passed",
        materialization_budget_status="accepted",
        realized_relative_l2=0.01,
        content_relative_l2_limit=3 / 250,
        actual_runtime_dtype="torch.float16",
    )


@pytest.mark.unit
def test_lf_whitened_directional_metric_keeps_failures_in_frozen_denominator() -> None:
    passing = aggregate_lf_whitened_direction(
        tuple(_metric_observation(index, passed=True) for index in range(32)),
    )
    scientific_negative = aggregate_lf_whitened_direction(
        tuple(_metric_observation(index, passed=False) for index in range(32)),
    )
    blocked = aggregate_lf_whitened_direction(
        tuple(_metric_observation(index, passed=True) for index in range(27)),
        implementation_failure_count=3,
        resource_failure_count=2,
    )

    assert passing.directional_validation_passed is True
    assert passing.module_outcome == "mechanism_signal_observed"
    assert passing.candidate_recommendation == "candidate_worth_further_selection"
    assert passing.registered_minus_max_wrong.exact_one_sided_confidence_lower_bound > 0.5
    assert scientific_negative.directional_validation_passed is False
    assert scientific_negative.module_outcome == "mechanism_signal_not_observed"
    assert scientific_negative.candidate_recommendation == (
        "candidate_not_recommended_for_selection"
    )
    assert blocked.directional_validation_passed is False
    assert blocked.module_outcome == "implementation_blocked"
    assert blocked.expected_cluster_count == 32
    assert blocked.successful_cluster_count == 27
    assert blocked.failed_cluster_count == 5
    assert blocked.implementation_failure_count == 3
    assert blocked.resource_failure_count == 2
    assert blocked.registered_minus_max_wrong.observation_count == 32
    assert blocked.registered_minus_max_wrong.practical_success_count == 27
    assert blocked.registered_minus_max_wrong.threshold_free_paired_ranking_auc == (
        27 / 32
    )
    assert (
        blocked.registered_minus_max_wrong.exact_one_sided_confidence_lower_bound
        == clopper_pearson_lower(27, 32, confidence_level=0.95)
    )


def _replace_scientific_record(
    record: DevelopmentScientificRecord,
    **changes: object,
) -> DevelopmentScientificRecord:
    provisional = replace(record, record_id="0" * 64, **changes)
    return replace(
        provisional,
        record_id=canonical_development_value_digest(
            provisional.payload_without_record_id()
        ),
    )


def _committed_marker(
    runner: LfWhitenedDirectionalValidationRunner,
    record: DevelopmentScientificRecord,
) -> CommittedUnit:
    record_bytes = (
        json.dumps(
            record.payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return CommittedUnit(
        schema_version="ceg_wm_development_committed_marker_v2",
        protocol_digest=runner.protocol_digest,
        revision=runner.method_code_revision,
        run_id=runner.run_id,
        shard_id="full",
        unit_id=str(record.analysis_unit_identity["unit_id"]),
        unit_index=record.unit_index,
        attempt_index=record.attempt_index,
        session_id="lf_whitened_directional_terminal_replay",
        fencing_token=1,
        intent_digest=canonical_digest(
            {"unit_index": record.unit_index, "attempt": record.attempt_index}
        ),
        attempt_disposition=record.attempt_disposition(),
        record_kind="development_scientific_record",
        record_id=record.record_id,
        record_digest=sha256(record_bytes).hexdigest(),
        record_bytes=len(record_bytes),
        actual_elapsed_seconds=float(record.actual_elapsed_seconds),
        maximum_duration_seconds=record.maximum_duration_seconds,
        bundle_sha256="2" * 64,
        bundle_bytes=1,
        artifact_manifest_digest="3" * 64,
        worker_identity_digest="4" * 64,
        parent_attempt_intent_digest=record.retry_parent_intent_digest,
        committed_at_utc="2026-08-10T00:00:00Z",
    )


def _terminal_failure_evidence(
    runner: LfWhitenedDirectionalValidationRunner,
    failure_classes: tuple[str, ...],
    *,
    first_failure_category: str | None = None,
) -> tuple[tuple[DevelopmentScientificRecord, CommittedUnit], ...]:
    evidence = []
    for ordinal, failure_class in enumerate(failure_classes):
        resource_failure = failure_class == "resource_failure"
        attempt_index = 1 if resource_failure else 0
        retry_parent = (
            canonical_digest({"resource_retry_parent": ordinal})
            if resource_failure
            else None
        )
        record = runner.create_failed_scientific_record(
            cluster_ordinal=ordinal,
            attempt_index=attempt_index,
            retry_parent_intent_digest=retry_parent,
            maximum_duration_seconds=2700,
            actual_elapsed_seconds=1.0,
            failure_type=(
                "builtins.MemoryError"
                if resource_failure
                else "builtins.RuntimeError"
            ),
            resource_failure=resource_failure,
            failure_category=(
                first_failure_category
                if ordinal == 0 and first_failure_category is not None
                else failure_class
            ),
        )
        evidence.append((record, _committed_marker(runner, record)))
    return tuple(evidence)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure_classes", "expected_outcome"),
    (
        (("implementation_failure",) * 32, "implementation_blocked"),
        (("resource_failure",) * 32, "resource_blocked"),
        (
            ("implementation_failure",) + ("resource_failure",) * 31,
            "implementation_blocked",
        ),
    ),
)
def test_lf_whitened_directional_replay_preserves_terminal_failure_outcome(
    failure_classes: tuple[str, ...],
    expected_outcome: str,
) -> None:
    runner, runtime = _directional_runner()
    evidence = _terminal_failure_evidence(
        runner,
        failure_classes,
        first_failure_category=(
            "identity_violation"
            if "implementation_failure" in failure_classes
            else None
        ),
    )

    aggregate = runner.replay_directional_aggregate(evidence)
    runtime.close()

    assert aggregate.successful_cluster_count == 0
    assert aggregate.failed_cluster_count == 32
    assert aggregate.implementation_failure_count == failure_classes.count(
        "implementation_failure"
    )
    assert aggregate.resource_failure_count == failure_classes.count(
        "resource_failure"
    )
    assert aggregate.module_outcome == expected_outcome
    assert aggregate.candidate_recommendation == (
        "candidate_not_recommended_for_selection"
    )
    assert aggregate.registered_minus_primary_null.observation_count == 32
    assert aggregate.registered_minus_primary_null.practical_success_count == 0
    assert aggregate.registered_minus_primary_null.threshold_free_paired_ranking_auc == 0.0
    assert aggregate.identity_violation_count == (
        1 if "implementation_failure" in failure_classes else 0
    )


@pytest.mark.unit
@pytest.mark.parametrize("failure_class", (None, "unregistered_failure"))
def test_lf_whitened_directional_replay_rejects_missing_or_unknown_failure_class(
    failure_class: str | None,
) -> None:
    runner, runtime = _directional_runner()
    evidence = list(
        _terminal_failure_evidence(runner, ("implementation_failure",) * 32)
    )
    invalid = _replace_scientific_record(
        evidence[0][0],
        failure_class=failure_class,
    )
    evidence[0] = (invalid, _committed_marker(runner, invalid))

    with pytest.raises(LfWhitenedDirectionalRunnerError):
        runner.replay_directional_aggregate(evidence)
    runtime.close()


@pytest.mark.unit
def test_lf_whitened_directional_replay_rejects_success_with_failure_class() -> None:
    runner, runtime = _directional_runner()
    evidence = list(
        _terminal_failure_evidence(runner, ("implementation_failure",) * 32)
    )
    invalid = _replace_scientific_record(
        evidence[0][0],
        execution_status="success",
        failure_reason=None,
    )
    evidence[0] = (invalid, _committed_marker(runner, invalid))

    with pytest.raises(LfWhitenedDirectionalRunnerError):
        runner.replay_directional_aggregate(evidence)
    runtime.close()


@pytest.mark.unit
def test_lf_whitened_directional_persistence_commits_recovers_and_preserves_retry_lineage(
    tmp_path: Path,
) -> None:
    runner, runtime = _directional_runner()
    identity = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest=runner.manifest.digest(),
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.run_id,
        worker_identity=identity,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="lf_directional_persistence_session",
        now_epoch_seconds=100,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=100)
    intent = store.create_session_intent(cursor, lease, now_epoch_seconds=101)
    record = _validated_operational_record(runner)
    store.commit_session_unit(
        cursor,
        lease,
        intent,
        record=record,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=102,
    )
    blocked_intent = store.create_session_intent(cursor, lease, now_epoch_seconds=103)
    blocked = runner.create_failed_scientific_record(
        cluster_ordinal=0,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.RuntimeError",
        resource_failure=False,
        failure_category="implementation_failure",
    )
    store.commit_session_unit(
        cursor,
        lease,
        blocked_intent,
        record=blocked,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=104,
    )
    retry_intent = store.create_session_intent(cursor, lease, now_epoch_seconds=105)
    retry_record = runner.create_failed_scientific_record(
        cluster_ordinal=1,
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.MemoryError",
        resource_failure=True,
        failure_category="resource_failure",
    )
    marker = store.commit_session_unit(
        cursor,
        lease,
        retry_intent,
        record=retry_record,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=106,
    )
    assert marker.attempt_disposition == "retryable_resource_failure"
    second_intent = store.create_session_intent(cursor, lease, now_epoch_seconds=107)
    assert second_intent.attempt_index == 1
    assert second_intent.parent_attempt_intent_digest == retry_intent.digest()
    terminal = runner.create_failed_scientific_record(
        cluster_ordinal=1,
        attempt_index=1,
        retry_parent_intent_digest=retry_intent.digest(),
        maximum_duration_seconds=2700,
        actual_elapsed_seconds=1.0,
        failure_type="builtins.MemoryError",
        resource_failure=True,
        failure_category="resource_failure",
    )
    store.commit_session_unit(
        cursor,
        lease,
        second_intent,
        record=terminal,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=108,
    )
    recovery = store.recover(now_epoch_seconds=109)
    runtime.close()

    assert cursor.next_unit_index == 3
    assert len(recovery.committed_units) == 4
    assert recovery.committed_units[-1].attempt_disposition == "final_failure"


@pytest.mark.integration
@pytest.mark.slow
def test_lf_whitened_directional_detector_record_persistence_full_chain(
    tmp_path: Path,
) -> None:
    runner, runtime = _directional_runner()
    identity = FrozenWorkerIdentity(
        revision=runner.method_code_revision,
        protocol_digest=runner.protocol_digest,
        execution_intent_authority_digest=runner.execution_intent_authority_digest,
        input_manifest_digest=runner.manifest.digest(),
        candidate_config_digest=runner.candidate_config_digest,
        unit_roster_digest=runner.protocol.unit_roster_digest,
    )
    store = DevelopmentPersistentStore(
        tmp_path,
        run_id=runner.run_id,
        worker_identity=identity,
        registered_unit_bindings=runner.create_persistence_unit_bindings(),
    )
    lease = store.acquire_lease(
        session_id="lf_directional_full_chain_session",
        now_epoch_seconds=200,
        lease_duration_seconds=10000,
    )
    cursor = store.open_session_cursor(lease, now_epoch_seconds=200)
    operational_intent = store.create_session_intent(
        cursor, lease, now_epoch_seconds=201
    )
    store.commit_session_unit(
        cursor,
        lease,
        operational_intent,
        record=_validated_operational_record(runner),
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=202,
    )
    scientific_intent = store.create_session_intent(
        cursor, lease, now_epoch_seconds=203
    )
    scientific = runner.execute_scientific_cluster(
        cluster_ordinal=0,
        base_latent=_lf_base_latent(),
        attempt_index=0,
        retry_parent_intent_digest=None,
        maximum_duration_seconds=2700,
    )
    committed = store.commit_session_unit(
        cursor,
        lease,
        scientific_intent,
        record=scientific,
        raw_secret_values=(ROOT_KEY, runner.registered_root_key),
        now_epoch_seconds=204,
    )
    recovery = store.recover(now_epoch_seconds=205)
    runtime.close()

    assert committed.attempt_disposition == "success"
    assert committed.record_id == scientific.record_id
    assert scientific.detector_trace["public_callable"] == (
        "main.lf_null_whitened_matched_detector"
    )
    assert scientific.detector_trace["same_image_registered_four_wrong_reuse"] is True
    assert scientific.detector_trace["paired_clean_primary_null"] is True
    assert cursor.next_unit_index == 2
    assert tuple(item.unit_index for item in recovery.committed_units) == (0, 1)
