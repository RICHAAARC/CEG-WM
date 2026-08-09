"""Frozen identity checks for LF whitened directional validation."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest

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
from scripts.experiment_execution.component_source_closure import (
    build_component_source_closure,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/lf_whitened_directional_validation.json"
READINESS = ROOT / ".codex/research_state/method_readiness.yaml"
SCREENING_CONFIG = ROOT / "configs/experiments/lf_whitened_score_screening.json"


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
def test_lf_whitened_directional_component_authority_replays_reviewed_sources() -> None:
    protocol, _manifest = load_lf_whitened_directional_validation_protocol(
        CONFIG, repository_root=ROOT
    )
    readiness = json.loads(READINESS.read_text("utf-8"))
    closure = build_component_source_closure(
        LF_DIRECTIONAL_COMPONENT_IDS,
        readiness["components"],
        ROOT,
    )

    assert protocol.ordered_component_ids == closure.ordered_component_ids
    assert tuple(asdict(item) for item in protocol.component_source_bindings) == (
        tuple(asdict(item) for item in closure.source_bindings)
    )
    assert (
        protocol.component_implementation_digest
        == closure.component_implementation_digest
    )
    assert protocol.candidate_specification_sha256 == sha256(
        (ROOT / "docs/design/candidate_specifications.md").read_bytes()
    ).hexdigest()
    assert protocol.method_review_reference == (
        readiness["independent_semantic_review"]["review_reference"]
    )
    assert protocol.method_reviewed_revision == (
        readiness["independent_semantic_review"][
            "reviewed_repository_revision"
        ]
    )


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
