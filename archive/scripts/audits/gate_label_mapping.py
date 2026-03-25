"""
File purpose: 审计编号标签映射与发布化描述统一入口。
Module type: General module
"""

from __future__ import annotations

from typing import Dict


_AUDIT_LABELS: Dict[str, Dict[str, str]] = {
    "B1.write_bypass_scan": {
        "audit_id": "records.write_path_is_unbypassable",
        "gate_name": "records.write_path_is_unbypassable",
        "legacy_code": "B1",
        "formal_description": "受控写盘路径必须不可旁路。",
    },
    "A6.yaml_loader_uniqueness": {
        "audit_id": "config.yaml_loader_is_safe_and_unique",
        "gate_name": "config.yaml_loader_is_safe_and_unique",
        "legacy_code": "A6",
        "formal_description": "YAML 解析必须安全且入口唯一。",
    },
    "C1.registry_injection_surface": {
        "audit_id": "registry.seal_and_runtime_injection_resistance",
        "gate_name": "registry.seal_and_runtime_injection_resistance",
        "legacy_code": "C1",
        "formal_description": "注册表必须 seal 且不可运行期注入。",
    },
    "B3_D1_D2.policy_path_semantics_binding": {
        "audit_id": "policy.path_semantics_binding_and_audit_evidence",
        "gate_name": "policy.path_semantics_binding_and_audit_evidence",
        "legacy_code": "B3_D1_D2",
        "formal_description": "policy_path 必须白名单绑定且保留路径审计证据。",
    },
    "D1.path_policy_escape_rejection": {
        "audit_id": "path.output_target_escape_rejection",
        "gate_name": "path.output_target_escape_rejection",
        "legacy_code": "D1",
        "formal_description": "输出路径必须拒绝逃逸并通过门禁校验。",
    },
    "D9.dangerous_exec_and_pickle_scan": {
        "audit_id": "runtime.dangerous_execution_and_deserialization_blocked",
        "gate_name": "runtime.dangerous_execution_and_deserialization_blocked",
        "legacy_code": "D9",
        "formal_description": "禁止危险动态执行与不安全反序列化。",
    },
    "D10.network_access_scan": {
        "audit_id": "runtime.network_access_is_audited_or_blocked",
        "gate_name": "runtime.network_access_is_audited_or_blocked",
        "legacy_code": "D10",
        "formal_description": "运行期网络访问必须受审计约束或阻断。",
    },
    "A1_A7.freeze_surface_integrity": {
        "audit_id": "freeze_surface.integrity_and_single_source_loading",
        "gate_name": "freeze_surface.integrity_and_single_source_loading",
        "legacy_code": "A1_A7",
        "formal_description": "冻结面完整性、唯一加载入口与锚定一致性。",
    },
    "S00.injection_scope_manifest_binding": {
        "audit_id": "injection_scope.manifest_binding_is_enforced",
        "gate_name": "injection_scope.manifest_binding_is_enforced",
        "legacy_code": "S00",
        "formal_description": "injection_scope_manifest 必须进入绑定事实源并受 schema 约束。",
    },
    "S01.records_schema_extensions_presence": {
        "audit_id": "records.schema_extensions_manifest_presence",
        "gate_name": "records.schema_extensions_manifest_presence",
        "legacy_code": "S01",
        "formal_description": "records_schema_extensions 配置必须存在且可解析。",
    },
    "S01.records_schema_extensions_append_only": {
        "audit_id": "records.schema_extensions_append_only_registration",
        "gate_name": "records.schema_extensions_append_only_registration",
        "legacy_code": "S01",
        "formal_description": "扩展字段必须 append-only 并登记到冻结注册表。",
    },
    "F1_F3_D2.evidence_bundle": {
        "audit_id": "evidence.run_root_bundle_is_complete_and_anchor_consistent",
        "gate_name": "evidence.run_root_bundle_is_complete_and_anchor_consistent",
        "legacy_code": "F1_F3_D2",
        "formal_description": "run_root 证据包必须完整且锚点一致。",
    },
}


def resolve_audit_label(legacy_audit_id: str, fallback_gate_name: str) -> Dict[str, str]:
    """
    功能：根据旧编号审计 ID 返回发布化标签。

    Resolve release-facing label fields from a legacy audit identifier.

    Args:
        legacy_audit_id: Legacy audit identifier, such as "B1.write_bypass_scan".
        fallback_gate_name: Fallback gate name when mapping is absent.

    Returns:
        Mapping containing audit_id, gate_name, legacy_code, and formal_description.
    """
    if not isinstance(legacy_audit_id, str) or not legacy_audit_id:
        raise TypeError("legacy_audit_id must be non-empty str")
    if not isinstance(fallback_gate_name, str) or not fallback_gate_name:
        raise TypeError("fallback_gate_name must be non-empty str")

    mapped = _AUDIT_LABELS.get(legacy_audit_id)
    if mapped is None:
        return {
            "audit_id": fallback_gate_name,
            "gate_name": fallback_gate_name,
            "legacy_code": legacy_audit_id,
            "formal_description": "未定义的审计标签映射，使用回退标识。",
        }
    return dict(mapped)
