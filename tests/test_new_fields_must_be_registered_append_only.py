"""
File purpose: 新增字段必须注册到 field_paths_registry 的审计回归测试。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.audits.audit_records_fields_append_only import run_audit


def test_new_fields_must_be_registered_append_only(tmp_path: Path) -> None:
    """
    功能：未登记扩展字段必须触发审计 FAIL。

    Missing registry binding for extension fields must fail append-only audit.

    Args:
        tmp_path: Temporary repository root path.

    Returns:
        None.
    """
    repo_root = tmp_path
    configs_dir = repo_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    extensions_yaml = """
version: \"v1\"
append_only: true
fields:
  - path: \"content_evidence.trajectory_evidence.planner_input_digest\"
    layer: \"anchor\"
    type: \"digest_hex64\"
    required: false
    missing_semantics: \"absent_ok\"
    description: \"planner input digest\"
""".strip()

    contracts_yaml = """
records_schema:
  field_paths_registry:
    - schema_version
    - content_evidence.trajectory_evidence
""".strip()

    (configs_dir / "records_schema_extensions.yaml").write_text(extensions_yaml, encoding="utf-8")
    (configs_dir / "frozen_contracts.yaml").write_text(contracts_yaml, encoding="utf-8")

    result = run_audit(repo_root)

    assert result["result"] == "FAIL"
    missing = result["evidence"]["missing_in_registry"]
    assert "content_evidence.trajectory_evidence.planner_input_digest" in missing


def test_negative_branch_attestation_provenance_registered_append_only_in_repo() -> None:
    """
    功能：验证 negative branch provenance 字段已在真实仓库中完成 append-only 注册。 

    Validate negative-branch provenance fields are append-only registered in
    the repository schema and frozen contracts.
    """
    repo_root = Path(__file__).resolve().parents[1]
    schema_obj = yaml.safe_load((repo_root / "configs" / "records_schema_extensions.yaml").read_text(encoding="utf-8"))
    contracts_obj = yaml.safe_load((repo_root / "configs" / "frozen_contracts.yaml").read_text(encoding="utf-8"))

    schema_fields = {
        entry.get("path")
        for entry in schema_obj.get("fields", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    registry_fields = set(contracts_obj.get("records_schema", {}).get("field_paths_registry", []))
    required_paths = {
        "negative_branch_source_attestation_provenance",
        "negative_branch_source_attestation_provenance.statement",
        "negative_branch_source_attestation_provenance.attestation_digest",
        "negative_branch_source_attestation_provenance.event_binding_digest",
        "negative_branch_source_attestation_provenance.trace_commit",
    }

    assert required_paths <= schema_fields
    assert required_paths <= registry_fields
