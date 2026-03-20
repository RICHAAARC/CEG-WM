"""
File purpose: attestation trace artifact 治理回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def _prepare_fact_sources(tmp_run_root: Path):
    from main.core.contracts import load_frozen_contracts
    from main.policy.runtime_whitelist import load_runtime_whitelist, load_policy_path_semantics
    from main.core.injection_scope import load_injection_scope_manifest

    contracts = load_frozen_contracts()
    whitelist = load_runtime_whitelist()
    semantics = load_policy_path_semantics()
    injection_scope_manifest = load_injection_scope_manifest()
    records_dir = tmp_run_root / "records"
    artifacts_dir = tmp_run_root / "artifacts"
    logs_dir = tmp_run_root / "logs"
    return contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir


def _build_lf_trace_payload() -> dict:
    return {
        "artifact_type": "lf_attestation_trace",
        "attestation_digest": "a" * 64,
        "event_binding_digest": "b" * 64,
        "trace_commit": "c" * 64,
        "lf_attestation_score": 0.5,
        "agreement_count": 18,
        "n_bits_compared": 36,
        "basis_rank": 36,
        "variance": 1.5,
        "edit_timestep": 0,
        "trajectory_feature_spec": {"feature_operator": "masked_normalized_random_projection", "edit_timestep": 0},
        "trajectory_feature_vector": [0.1, 0.2, 0.3],
        "trajectory_feature_digest": "d" * 64,
        "projected_lf_coeffs": [0.2, -0.1, 0.05],
        "projected_lf_signs": [1, -1, 1],
        "projected_lf_digest": "e" * 64,
        "expected_bit_signs": [1, -1, 1],
        "posterior_values": [0.2, -0.1, 0.05],
        "posterior_signs": [1, -1, 1],
        "posterior_margin_values": [0.2, 0.1, 0.05],
        "agreement_indices": [0, 1, 2],
        "mismatch_indices": [],
        "weakest_posterior_indices": [2, 1],
        "weakest_posterior_margins": [0.05, 0.1],
        "plan_digest": "f" * 64,
        "lf_basis_digest": "1" * 64,
        "projection_matrix_digest": "2" * 64,
        "trajectory_feature_spec_digest": "3" * 64,
        "projection_seed": 17,
        "lf_attestation_trace_digest": "4" * 64,
    }


def test_attestation_trace_artifact_contracts_are_registered_append_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contracts_obj = yaml.safe_load((repo_root / "configs" / "frozen_contracts.yaml").read_text(encoding="utf-8"))
    schema_obj = yaml.safe_load((repo_root / "configs" / "records_schema_extensions.yaml").read_text(encoding="utf-8"))

    artifact_schema = contracts_obj.get("artifact_schema")
    assert artifact_schema.get("append_only") is True
    artifact_contracts = artifact_schema.get("artifact_contracts")
    assert "lf_attestation_trace" in artifact_contracts
    assert "hf_attestation_trace" in artifact_contracts

    lf_allowed_fields = set(artifact_contracts["lf_attestation_trace"]["allowed_top_level_fields"])
    required_lf_fields = {
        "expected_bit_signs",
        "posterior_values",
        "posterior_signs",
        "posterior_margin_values",
        "agreement_indices",
        "mismatch_indices",
        "weakest_posterior_indices",
        "weakest_posterior_margins",
        "projected_lf_coeffs",
        "projected_lf_signs",
        "trajectory_feature_vector",
        "plan_digest",
        "lf_basis_digest",
        "projection_matrix_digest",
        "trajectory_feature_spec_digest",
        "projection_seed",
    }
    assert required_lf_fields <= lf_allowed_fields

    registry_fields = set(contracts_obj.get("records_schema", {}).get("field_paths_registry", []))
    schema_fields = {
        entry.get("path")
        for entry in schema_obj.get("fields", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    assert "expected_bit_signs" not in registry_fields
    assert "posterior_values" not in registry_fields
    assert "expected_bit_signs" not in schema_fields
    assert "posterior_values" not in schema_fields


def test_lf_attestation_trace_artifact_rejects_unallowlisted_top_level_field(tmp_run_root) -> None:
    from main.core import records_io
    from main.core.errors import RecordsWritePolicyError

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)
    payload = _build_lf_trace_payload()
    payload["illegal_extra_field"] = "not allowed"
    output_path = artifacts_dir / "attestation" / "lf_attestation_trace.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        with pytest.raises(RecordsWritePolicyError) as exc_info:
            records_io.write_artifact_json(str(output_path), payload)

    assert "allowlist" in str(exc_info.value).lower()


def test_lf_attestation_trace_artifact_accepts_governed_field_set(tmp_run_root) -> None:
    from main.core import records_io

    contracts, whitelist, semantics, injection_scope_manifest, records_dir, artifacts_dir, logs_dir = _prepare_fact_sources(tmp_run_root)
    payload = _build_lf_trace_payload()
    output_path = artifacts_dir / "attestation" / "lf_attestation_trace.json"

    with records_io.bound_fact_sources(
        contracts,
        whitelist,
        semantics,
        tmp_run_root,
        records_dir,
        artifacts_dir,
        logs_dir,
        injection_scope_manifest=injection_scope_manifest,
    ):
        records_io.write_artifact_json(str(output_path), payload)

    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload["artifact_type"] == "lf_attestation_trace"
    assert written_payload["expected_bit_signs"] == [1, -1, 1]
    assert written_payload["_artifact_audit"]["writer"] == "records_io"