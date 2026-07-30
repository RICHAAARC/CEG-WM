"""C1 HF-only reference protocol, frozen roster, and budget constraints."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import shutil

import pytest

from experiments.protocol.c1_hf_reference import (
    C1_HF_CONFIDENCE_LEVEL,
    C1_HF_DATASET_SHA256,
    C1_HF_MANIFEST_IDENTITIES,
    C1_HF_PRIMARY_NULL_TARGET_FPR,
    C1_HF_PROMPT_COUNT,
    C1_HF_PROMPTS_PER_SPLIT,
    C1_HF_SOURCE_CLUSTERS_PER_SPLIT,
    C1_HF_THRESHOLD_TAIL_FAILURE_PROBABILITY,
    C1_HF_ZERO_FAILURE_CP_UPPER_95,
    load_c1_hf_reference_bundle,
    materialize_c1_split_manifest,
    validate_c1_manifest_pair,
)
from experiments.protocol.internal_splits import (
    INTERNAL_VALIDATION_PROTOCOL_ID,
    INTERNAL_VALIDATION_PROTOCOL_VERSION,
    LEGACY_INTERNAL_VALIDATION_PROTOCOL_ID,
    LEGACY_INTERNAL_VALIDATION_PROTOCOL_VERSION,
)
from experiments.protocol.internal_validation import LEGACY_PROTOCOL_COMPATIBILITY


ROOT = Path(__file__).resolve().parents[2]
C1_MODULE = ROOT / "experiments/protocol/c1_hf_reference.py"


@pytest.mark.unit
def test_c1_bundle_loads_exact_authorities_and_offline_prompt_snapshot() -> None:
    bundle = load_c1_hf_reference_bundle(ROOT)
    specification = bundle.specification.raw
    assert specification["protocol_id"] == INTERNAL_VALIDATION_PROTOCOL_ID
    assert specification["protocol_version"] == INTERNAL_VALIDATION_PROTOCOL_VERSION
    assert specification["dataset"]["runtime_network_access"] == (
        "forbidden_use_frozen_roster_only"
    )
    assert len(bundle.roster.rows) == C1_HF_PROMPT_COUNT
    assert len({row.prompt_text for row in bundle.roster.rows}) == C1_HF_PROMPT_COUNT
    assert len({row.prompt_digest for row in bundle.roster.rows}) == C1_HF_PROMPT_COUNT
    assert {row.source_row for row in bundle.roster.rows} == set(
        range(1, C1_HF_PROMPT_COUNT + 1)
    )
    assert all(
        hashlib.sha256(row.prompt_text.encode("utf-8")).hexdigest()
        == row.prompt_digest
        for row in bundle.roster.rows
    )
    snapshot = ROOT / specification["dataset"]["dataset_snapshot_path"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == C1_HF_DATASET_SHA256


@pytest.mark.unit
def test_c1_bundle_rejects_candidate_specification_authority_tamper(tmp_path) -> None:
    bundle = load_c1_hf_reference_bundle(ROOT)
    specification = bundle.specification.raw
    binding = specification["candidate_binding"]
    temporary_root = tmp_path / "repository"
    bound_paths = {
        Path("configs/experiments/c1_hf_reference_run.json"),
        Path(binding["candidate_specification_path"]),
        Path(binding["formal_method_adapter_config_path"]),
        Path(binding["runtime_config_path"]),
        Path(specification["dataset"]["roster_path"]),
        Path(specification["dataset"]["dataset_snapshot_path"]),
        *(
            Path(manifest["path"])
            for manifest in specification["split_manifests"].values()
        ),
        *(Path(entry["path"]) for entry in binding["method_source_files"]),
    }
    for relative_path in bound_paths:
        destination = temporary_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)

    candidate_specification = temporary_root / binding["candidate_specification_path"]
    candidate_specification.write_bytes(
        candidate_specification.read_bytes() + b"\nauthority tamper\n"
    )

    with pytest.raises(
        ValueError,
        match="^candidate_specification_authority_mismatch$",
    ):
        load_c1_hf_reference_bundle(temporary_root)


@pytest.mark.unit
def test_c1_compact_manifests_materialize_deterministically_and_are_disjoint() -> None:
    bundle = load_c1_hf_reference_bundle(ROOT)
    threshold_fit, confirmation = bundle.materialized_manifests
    assert tuple(len(manifest.assignments) for manifest in bundle.materialized_manifests) == (
        C1_HF_SOURCE_CLUSTERS_PER_SPLIT,
        C1_HF_SOURCE_CLUSTERS_PER_SPLIT,
    )
    assert tuple(manifest.digest() for manifest in bundle.materialized_manifests) == (
        C1_HF_MANIFEST_IDENTITIES["content_threshold_fit"][2],
        C1_HF_MANIFEST_IDENTITIES["untouched_confirmation"][2],
    )
    assert tuple(
        materialize_c1_split_manifest(compact, bundle.roster).digest()
        for compact in bundle.compact_manifests
    ) == tuple(
        manifest.digest() for manifest in bundle.materialized_manifests
    )
    assert validate_c1_manifest_pair(
        threshold_fit,
        confirmation,
        bundle.roster,
    ) == ()
    threshold_prompts = {
        assignment.identity.prompt_digest
        for assignment in threshold_fit.assignments
    }
    confirmation_prompts = {
        assignment.identity.prompt_digest for assignment in confirmation.assignments
    }
    assert len(threshold_prompts) == len(confirmation_prompts) == C1_HF_PROMPTS_PER_SPLIT
    assert not threshold_prompts & confirmation_prompts
    assert threshold_prompts | confirmation_prompts == {
        row.prompt_digest for row in bundle.roster.rows
    }


@pytest.mark.unit
def test_c1_prompt_split_is_balanced_within_every_category_challenge_stratum() -> None:
    bundle = load_c1_hf_reference_bundle(ROOT)
    threshold_fit, confirmation = bundle.materialized_manifests
    roster_by_digest = {row.prompt_digest: row for row in bundle.roster.rows}

    def stratum_counts(manifest) -> Counter[tuple[str, str]]:
        prompt_digests = {
            assignment.identity.prompt_digest for assignment in manifest.assignments
        }
        return Counter(
            (
                roster_by_digest[digest].category,
                roster_by_digest[digest].challenge,
            )
            for digest in prompt_digests
        )

    threshold_counts = stratum_counts(threshold_fit)
    confirmation_counts = stratum_counts(confirmation)
    strata = {
        (row.category, row.challenge) for row in bundle.roster.rows
    }
    assert all(
        abs(threshold_counts[stratum] - confirmation_counts[stratum]) <= 1
        for stratum in strata
    )


@pytest.mark.unit
def test_c1_compact_manifest_identity_and_materialized_digest_fail_closed() -> None:
    bundle = load_c1_hf_reference_bundle(ROOT)
    compact = bundle.compact_manifests[0]
    assert "manifest_id_frozen_value_mismatch" in replace(
        compact,
        manifest_id="forged",
    ).validate()
    assert "manifest_revision_frozen_value_mismatch" in replace(
        compact,
        manifest_revision="2",
    ).validate()
    assert "expected_materialized_manifest_digest_frozen_value_mismatch" in replace(
        compact,
        expected_materialized_manifest_digest="f" * 64,
    ).validate()


@pytest.mark.unit
def test_c1_reference_freeze_is_authority_fact_not_result_gate() -> None:
    bundle = load_c1_hf_reference_bundle(ROOT)
    frozen = bundle.specification.freeze_reference_candidate()
    assert frozen.gate_id == "hf_reference_candidate_frozen"
    assert frozen.detector_mode == "hf_only"
    assert frozen.freeze_semantics.endswith("not_result_gate")
    required = bundle.specification.raw["candidate_binding"][
        "required_execution_package_bindings"
    ]
    assert required["absence_semantics"] == "c1_execution_preflight_fail_closed"
    assert required["c1_protocol_status"] == "not_yet_materialized_no_result_claim"


@pytest.mark.unit
def test_c1_run_phases_cannot_mix_fit_and_confirmation() -> None:
    specification = load_c1_hf_reference_bundle(ROOT).specification.raw
    phases = specification["run_phases"]
    assert phases["threshold_fit"]["accessible_split"] == "content_threshold_fit"
    assert phases["threshold_fit"]["forbidden_split_access"] == [
        "untouched_confirmation"
    ]
    assert phases["threshold_fit"]["same_run_confirmation"] == "forbidden"
    assert phases["untouched_confirmation"]["prerequisite_gates"] == [
        "candidate_selection_frozen",
        "hf_only_tau_frozen",
    ]
    assert phases["untouched_confirmation"]["tau_refit"] == "forbidden"
    assert phases["untouched_confirmation"]["package_authorization_status"].startswith(
        "blocked_until_"
    )


@pytest.mark.unit
def test_c1_statistics_and_workload_are_independent_and_falsifiable() -> None:
    specification = load_c1_hf_reference_bundle(ROOT).specification.raw
    statistics = specification["statistics"]
    assert C1_HF_THRESHOLD_TAIL_FAILURE_PROBABILITY == pytest.approx(
        (1.0 - C1_HF_PRIMARY_NULL_TARGET_FPR) ** C1_HF_SOURCE_CLUSTERS_PER_SPLIT
    )
    assert C1_HF_THRESHOLD_TAIL_FAILURE_PROBABILITY < 0.05
    assert C1_HF_ZERO_FAILURE_CP_UPPER_95 == pytest.approx(
        1.0
        - (1.0 - C1_HF_CONFIDENCE_LEVEL)
        ** (1.0 / C1_HF_SOURCE_CLUSTERS_PER_SPLIT)
    )
    assert C1_HF_ZERO_FAILURE_CP_UPPER_95 <= C1_HF_PRIMARY_NULL_TARGET_FPR
    assert statistics["threshold_fit"]["fit_rule"].startswith("float64_nextafter")
    assert statistics["threshold_fit"]["confirmation_data_access"] == "forbidden"
    assert statistics["untouched_confirmation"]["tau_refit"] == "forbidden"
    assert statistics["tpr"]["minimum_lower_confidence_bound"] == 0.5
    assert statistics["wrong_key"]["maximum_upper_confidence_bound"] == 0.001
    assert statistics["wrong_key"]["reported_separately_from_primary_null"]
    assert statistics["paired_attribution"]["ties"] == "count_as_failures"
    result_gate = statistics["result_gate_semantics"]
    assert result_gate["result_gate_is_prerequisite"] is False
    assert "forbids_c2" in result_gate["negative_closure"]

    budget = specification["execution_budget"]
    assert budget["threshold_fit"]["total_detection_calls"] == 4096
    assert budget["untouched_confirmation"]["total_detection_calls"] == 12288
    assert budget["failure_and_denominator_policy"]["early_stopping"] == "forbidden"
    assert budget["shard_and_resource_boundary"]["shard_count_per_phase"] * (
        budget["shard_and_resource_boundary"]["source_clusters_per_shard"]
    ) == 4096
    assert not budget["shard_and_resource_boundary"][
        "gpu_execution_authorized_by_c1_protocol"
    ]


@pytest.mark.unit
def test_c1_metric_identities_are_split_bound_without_protocol_implementation() -> None:
    specification = load_c1_hf_reference_bundle(ROOT).specification.raw
    metric_plan = specification["metric_plan"]
    bindings = {
        item["metric_id"]: item["allowed_splits"]
        for item in metric_plan["metric_split_bindings"]
    }
    assert bindings["c1_hf_tau_fit"] == ["content_threshold_fit"]
    assert all(
        splits == ["untouched_confirmation"]
        for metric_id, splits in bindings.items()
        if metric_id != "c1_hf_tau_fit"
    )
    assert all("held_out_evaluation" not in splits for splits in bindings.values())
    quality = metric_plan["formula_identities"]["paired_quality"]
    assert quality["reference"] == "final_clean_rgb8_image"
    assert quality["candidate"].startswith("same_source_cluster_final_registered")
    assert quality["required_pair_count"] == 4096
    assert quality["pass_cutoff"] is None
    assert quality["missing_or_non_finite"].endswith("gate_failure")

    module_tree = ast.parse(C1_MODULE.read_text(encoding="utf-8"))
    function_names = {
        node.name for node in ast.walk(module_tree) if isinstance(node, ast.FunctionDef)
    }
    assert "fit_hf_only_threshold" not in function_names
    assert "clopper_pearson" not in " ".join(function_names)


@pytest.mark.unit
def test_protocol_upgrade_rejects_legacy_semantic_reinterpretation() -> None:
    assert INTERNAL_VALIDATION_PROTOCOL_ID.endswith("_v2")
    assert INTERNAL_VALIDATION_PROTOCOL_VERSION == "2.0.0"
    assert LEGACY_INTERNAL_VALIDATION_PROTOCOL_ID.endswith("_v1")
    assert LEGACY_INTERNAL_VALIDATION_PROTOCOL_VERSION == "1.0.0"
    assert LEGACY_PROTOCOL_COMPATIBILITY == (
        "v1_structure_readable_but_semantically_incompatible_and_not_revalidatable_as_v2"
    )
