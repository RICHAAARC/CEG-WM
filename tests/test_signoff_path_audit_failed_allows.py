"""
功能：测试 path_audit 失败时签署决策不应被阻断

Module type: Core innovation module

Test that signoff allows freeze when path_audit_status is failed
and required evidence exists.
"""

import json
from pathlib import Path


def test_signoff_allows_when_path_audit_failed(tmp_run_root: Path):
    """
    Test signoff decision when path_audit_status is failed.

    当 path_audit_status="failed" 且证据齐全时，签署决策应为 ALLOW_FREEZE。
    """
    from main.core import time_utils
    from scripts.run_freeze_signoff import validate_run_root_evidence, compute_signoff_decision

    artifacts_dir = tmp_run_root / "artifacts"
    cfg_audit_path = artifacts_dir / "cfg_audit" / "cfg_audit.json"
    cfg_audit_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_audit_path.write_text(json.dumps({"cfg_digest": "test"}, indent=2), encoding="utf-8")

    env_audits_dir = artifacts_dir / "env_audits"
    env_audits_dir.mkdir(parents=True, exist_ok=True)
    env_audit_path = env_audits_dir / "env_audit_test.json"
    env_audit_path.write_text(json.dumps({"generated_at_utc": time_utils.now_utc_iso_z()}), encoding="utf-8")

    run_closure_path = artifacts_dir / "run_closure.json"
    run_closure_payload = {
        "schema_version": "v1.0",
        "run_id": "test_signoff_001",
        "command": "unit_test",
        "created_at_utc": time_utils.now_utc_iso_z(),
        "cfg_digest": "cfg_digest_test",
        "contract_version": "<absent>",
        "contract_digest": "<absent>",
        "contract_file_sha256": "<absent>",
        "contract_canon_sha256": "<absent>",
        "contract_bound_digest": "<absent>",
        "whitelist_version": "<absent>",
        "whitelist_digest": "<absent>",
        "whitelist_file_sha256": "<absent>",
        "whitelist_canon_sha256": "<absent>",
        "whitelist_bound_digest": "<absent>",
        "policy_path_semantics_version": "<absent>",
        "policy_path_semantics_digest": "<absent>",
        "policy_path_semantics_file_sha256": "<absent>",
        "policy_path_semantics_canon_sha256": "<absent>",
        "policy_path_semantics_bound_digest": "<absent>",
        "policy_path": "test_policy_path",
        "impl_id": "test_impl",
        "impl_version": "1.0.0",
        "impl_identity": None,
        "impl_identity_digest": None,
        "facts_anchor": None,
        "records_bundle": None,
        "status": {
            "ok": False,
            "reason": "runtime_error",
            "details": {
                "note": "path_audit_unbound"
            }
        },
        "path_audit_status": "failed",
        "path_audit_error_code": "fact_sources_unbound",
        "path_audit_error": "FactSourcesNotInitializedError: bound_fact_sources missing"
    }
    run_closure_path.write_text(json.dumps(run_closure_payload, indent=2), encoding="utf-8")

    evidence_report = validate_run_root_evidence(tmp_run_root)
    assert evidence_report.get("status") == "ok"

    decision = compute_signoff_decision(static_audits=[], evidence_report=evidence_report)
    assert decision.get("decision") == "ALLOW_FREEZE"
