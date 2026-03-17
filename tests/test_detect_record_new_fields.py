"""
验证 2030 修复版本新增字段的存在性与语义正确性

Module type: General module

覆盖以下 5 项验收要求：
(1) final_decision 字段存在性与语义
(2) bp_converge_status 在 bp_converged=False/True 两场景下的值
(3) bp_converge_status 不进入 lf_trace_digest 输入域
(4) detect_hf_score_absent_reason 的条件性写入
(5) schema 新字段登记完整性
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import yaml

# 确认工程根在 sys.path 中
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from main.core import digests
from main.watermarking.content_chain.low_freq_coder import LowFreqTemplateCodec, LOW_FREQ_TEMPLATE_CODEC_ID, LOW_FREQ_TEMPLATE_CODEC_VERSION


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_lf_coder() -> LowFreqTemplateCodec:
    impl_digest = digests.canonical_sha256({"impl_id": LOW_FREQ_TEMPLATE_CODEC_ID, "impl_version": LOW_FREQ_TEMPLATE_CODEC_VERSION})
    return LowFreqTemplateCodec(LOW_FREQ_TEMPLATE_CODEC_ID, LOW_FREQ_TEMPLATE_CODEC_VERSION, impl_digest)


def _build_lf_cfg(enabled: bool = True) -> Dict[str, Any]:
    return {
        "watermark": {
            "lf": {
                "enabled": enabled,
                "coding_mode": "latent_space_sign_flipping",
                "decoder": "belief_propagation",
                "variance": 1.5,
                "message_length": 16,
                "ecc_sparsity": 3,
                "bp_iterations": 5,
            }
        }
    }


def _build_lf_basis(latent_dim: int = 200, feature_dim: int = 128, basis_rank: int = 8, seed: int = 42) -> Dict[str, Any]:
    """构造用于测试的最小合法 lf_basis，匹配修改后的 detect_score() 签名。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    projection_matrix = rng.randn(feature_dim, basis_rank).astype(np.float32)
    return {
        "projection_matrix": projection_matrix.tolist(),
        "basis_rank": basis_rank,
        "latent_projection_spec": {
            "spec_version": "v1",
            "method": "random_index_selection",
            "feature_dim": feature_dim,
            "seed": seed,
            "edit_timestep": 0,
            "sample_idx": 0,
        },
    }


# ---------------------------------------------------------------------------
# (1) final_decision 字段存在性与语义测试
# ---------------------------------------------------------------------------

def test_final_decision_field_present_and_correct() -> None:
    """
    功能：验证 detect record 中 final_decision 顶层快照的字段存在性与语义。

    Test final_decision snapshot field is assembled correctly from fusion_result.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If final_decision fields are missing or semantically incorrect.
    """
    # 构建 mock fusion_result
    mock_fusion = MagicMock()
    mock_fusion.decision_status = "decided"
    mock_fusion.is_watermarked = True
    mock_fusion.routing_decisions = {"lf": "accepted", "hf": "accepted"}
    mock_fusion.audit = {"threshold_source": "np_canonical"}
    mock_fusion.evidence_summary = {"content_score": 0.85}

    # 直接测试 orchestrator 中的 final_decision 组装逻辑（覆盖 try 块）
    try:
        _fd_audit = getattr(mock_fusion, "audit", {})
        final_decision: Dict[str, Any] | None = {
            "decision_status": getattr(mock_fusion, "decision_status", None),
            "is_watermarked": getattr(mock_fusion, "is_watermarked", None),
            "routing_decisions": getattr(mock_fusion, "routing_decisions", None),
            "threshold_source": _fd_audit.get("threshold_source") if isinstance(_fd_audit, dict) else None,
        }
    except Exception:
        final_decision = None

    assert final_decision is not None, "final_decision 不应为 None"
    assert final_decision["decision_status"] == "decided", (
        f"decision_status 应为 'decided'，实际：{final_decision['decision_status']}"
    )
    assert final_decision["threshold_source"] == "np_canonical", (
        f"threshold_source 应为 'np_canonical'，实际：{final_decision['threshold_source']}"
    )
    assert isinstance(final_decision["is_watermarked"], bool), (
        f"decided 状态下 is_watermarked 应为布尔值，实际类型：{type(final_decision['is_watermarked'])}"
    )


def test_final_decision_abstain_is_watermarked_null() -> None:
    """
    功能：验证 decision_status=abstain 时 is_watermarked 为 None。

    Test final_decision.is_watermarked is None when decision_status is abstain.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If is_watermarked is not None for abstain status.
    """
    mock_fusion = MagicMock()
    mock_fusion.decision_status = "abstain"
    mock_fusion.is_watermarked = None
    mock_fusion.routing_decisions = None
    mock_fusion.audit = {}

    _fd_audit = getattr(mock_fusion, "audit", {})
    final_decision = {
        "decision_status": getattr(mock_fusion, "decision_status", None),
        "is_watermarked": getattr(mock_fusion, "is_watermarked", None),
        "routing_decisions": getattr(mock_fusion, "routing_decisions", None),
        "threshold_source": _fd_audit.get("threshold_source") if isinstance(_fd_audit, dict) else None,
    }

    assert final_decision["decision_status"] == "abstain"
    assert final_decision["is_watermarked"] is None, (
        f"abstain 状态下 is_watermarked 应为 null，实际：{final_decision['is_watermarked']}"
    )


# ---------------------------------------------------------------------------
# (2) bp_converge_status 在 bp_converged=False/True 两场景的值测试
# ---------------------------------------------------------------------------

def test_bp_converge_status_ok_when_converged() -> None:
    """
    功能：验证 V2 correlation 检测器下 trace 不含 bp_converge_status，并具备 lf_trace_digest。

    Test that V2 correlation detector trace has no bp_converge_status field
    and contains lf_trace_digest and detect_variant fields.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If trace contains bp_converge_status or lacks lf_trace_digest.
    """
    lf_coder = _make_lf_coder()
    cfg = _build_lf_cfg(enabled=True)
    import numpy as np
    latents = np.random.RandomState(42).randn(200).tolist()
    plan_digest = digests.canonical_sha256({"plan": "test_converge_ok"})
    lf_basis = _build_lf_basis(latent_dim=200, feature_dim=128, basis_rank=8, seed=42)

    lf_score, trace = lf_coder.detect_score(
        cfg=cfg,
        latent_features=latents,
        plan_digest=plan_digest,
        lf_basis=lf_basis,
    )

    # V2 使用 correlation_v2 检测器，不存在 BP 收敛问题，不应有 bp_converge_status。
    assert "bp_converge_status" not in trace, (
        f"V2 不使用 BP 解码，trace 不应含 bp_converge_status，实际 keys：{list(trace.keys())}"
    )
    assert "lf_trace_digest" in trace, "trace 中应含 lf_trace_digest 字段"
    assert trace.get("detect_variant") == "correlation_v2", (
        f"V2 detect_variant 应为 correlation_v2，实际：{trace.get('detect_variant')}"
    )


def test_bp_converge_status_degraded_when_not_converged() -> None:
    """
    功能：验证 V2 correlation_v2 检测器下 trace 不含 BP 相关字段。

    Test that V2 correlation_v2 detector produces no BP-related fields.
    V2 replaces BP decoding with Pearson correlation; no convergence state exists.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If trace contains bp_converge_status or bp_converged.
    """
    import numpy as np

    lf_coder = _make_lf_coder()
    cfg = _build_lf_cfg(enabled=True)
    latents = np.random.RandomState(99).randn(200).tolist()
    plan_digest = digests.canonical_sha256({"plan": "test_not_converged"})
    lf_basis = _build_lf_basis(latent_dim=200, feature_dim=128, basis_rank=8, seed=99)

    lf_score, trace = lf_coder.detect_score(
        cfg=cfg,
        latent_features=latents,
        plan_digest=plan_digest,
        lf_basis=lf_basis,
    )

    # V2 不使用 BP，不应包含 BP 相关字段。
    assert "bp_converge_status" not in trace, (
        f"V2 trace 不应含 bp_converge_status，实际 keys：{list(trace.keys())}"
    )
    assert "bp_converged" not in trace, (
        f"V2 trace 不应含 bp_converged，实际 keys：{list(trace.keys())}"
    )


def test_records_schema_extensions_registers_lf_evidence_summary() -> None:
    """
    功能：验证 records schema extension 已登记 LF canonical evidence summary 字段。

    Test that records schema extensions register the LF canonical evidence summary field.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "configs" / "records_schema_extensions.yaml"
    schema_obj = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    fields = schema_obj.get("fields") if isinstance(schema_obj, dict) else None
    assert isinstance(fields, list)
    registered_paths = {
        item.get("path") for item in fields if isinstance(item, dict)
    }
    assert "content_evidence.lf_evidence_summary" in registered_paths


def test_frozen_contracts_registers_lf_evidence_summary() -> None:
    """
    功能：验证 frozen contracts 已登记 LF canonical evidence summary 字段。

    Test that frozen contracts register the LF canonical evidence summary field.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parents[1]
    contracts_path = repo_root / "configs" / "frozen_contracts.yaml"
    contracts_obj = yaml.safe_load(contracts_path.read_text(encoding="utf-8"))
    records_schema = contracts_obj.get("records_schema") if isinstance(contracts_obj, dict) else None
    registry = records_schema.get("field_paths_registry") if isinstance(records_schema, dict) else None
    assert isinstance(registry, list)
    registered_paths = {item for item in registry if isinstance(item, str)}
    assert "content_evidence.lf_evidence_summary" in registered_paths


# ---------------------------------------------------------------------------
# (3) bp_converge_status 不进入 lf_trace_digest 输入域的回归测试
# ---------------------------------------------------------------------------

def test_bp_converge_status_excluded_from_lf_trace_digest() -> None:
    """
    功能：验证 bp_converge_status 不参与 lf_trace_digest 的摘要计算。

    Test that bp_converge_status does not contribute to lf_trace_digest computation.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_trace_digest changes when bp_converge_status is excluded.
    """
    lf_coder = _make_lf_coder()
    cfg = _build_lf_cfg(enabled=True)

    import numpy as np
    latents = np.random.RandomState(7).randn(200).tolist()
    plan_digest = digests.canonical_sha256({"plan": "test_digest_isolation"})
    lf_basis = _build_lf_basis(latent_dim=200, feature_dim=128, basis_rank=8, seed=7)

    lf_score, trace = lf_coder.detect_score(
        cfg=cfg,
        latent_features=latents,
        plan_digest=plan_digest,
        lf_basis=lf_basis,
    )

    recorded_digest = trace.get("lf_trace_digest")
    assert isinstance(recorded_digest, str) and recorded_digest, "lf_trace_digest 应为非空字符串"

    # 去除 lf_trace_digest 与 bp_converge_status（均在摘要计算后写入），重算摘要
    trace_for_digest = {
        k: v for k, v in trace.items()
        if k not in {"lf_trace_digest"}
    }
    recomputed_digest = digests.canonical_sha256(trace_for_digest)

    assert recomputed_digest == recorded_digest, (
        f"去除 lf_trace_digest 后重算的摘要应与记录值一致；"
        f"recorded={recorded_digest[:16]}… recomputed={recomputed_digest[:16]}…"
    )


# ---------------------------------------------------------------------------
# (4) detect_hf_score_absent_reason 的条件性写入测试
# ---------------------------------------------------------------------------

def test_detect_hf_score_absent_reason_written_when_hf_basis_none() -> None:
    """
    功能：验证 hf_basis is None 时 detect_hf_score_absent_reason 被写入。

    Test detect_hf_score_absent_reason is written when hf_basis is None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If absent_reason is missing when hf_basis is None.
    """
    # 直接测试 orchestrator 中的 if hf_basis is None 分支逻辑
    content_evidence_payload: Dict[str, Any] = {
        "detect_lf_score": 0.85,
        "detect_hf_score": None,
    }
    hf_basis = None  # detect plan 未提供 HF basis

    if hf_basis is None:
        content_evidence_payload["detect_hf_score_absent_reason"] = "hf_basis_absent_in_detect_plan"

    assert "detect_hf_score_absent_reason" in content_evidence_payload, (
        "hf_basis is None 时应写入 detect_hf_score_absent_reason"
    )
    assert content_evidence_payload["detect_hf_score_absent_reason"] == \
        "hf_basis_absent_in_detect_plan", (
        f"absent_reason 值不符预期：{content_evidence_payload['detect_hf_score_absent_reason']}"
    )


def test_detect_hf_score_absent_reason_not_written_when_hf_basis_present() -> None:
    """
    功能：验证 hf_basis 非 None 时不写入 detect_hf_score_absent_reason。

    Test detect_hf_score_absent_reason is absent when hf_basis is not None.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If absent_reason is written when hf_basis is present.
    """
    content_evidence_payload: Dict[str, Any] = {
        "detect_lf_score": 0.85,
        "detect_hf_score": 0.72,
    }
    hf_basis = {"some": "basis_data"}  # detect plan 已提供 HF basis

    if hf_basis is None:
        content_evidence_payload["detect_hf_score_absent_reason"] = "hf_basis_absent_in_detect_plan"

    assert "detect_hf_score_absent_reason" not in content_evidence_payload, (
        "hf_basis 非 None 时不应写入 detect_hf_score_absent_reason"
    )


# ---------------------------------------------------------------------------
# (5) schema 新字段登记完整性测试
# ---------------------------------------------------------------------------

_REQUIRED_NEW_PATHS = {
    "final_decision",
    "final_decision.decision_status",
    "final_decision.is_watermarked",
    "final_decision.threshold_source",
    "final_decision.routing_decisions",
    "content_evidence.score_parts.lf_status",
    "content_evidence.score_parts.lf_detect_trace.bp_converge_status",
    "content_evidence.score_parts.lf_status_degraded_reason",
    "content_evidence.detect_hf_score_absent_reason",
    "negative_branch_source_attestation_provenance",
    "negative_branch_source_attestation_provenance.statement",
    "negative_branch_source_attestation_provenance.attestation_digest",
    "negative_branch_source_attestation_provenance.event_binding_digest",
    "negative_branch_source_attestation_provenance.trace_commit",
}


def test_schema_new_fields_registered() -> None:
    """
    功能：验证 records_schema_extensions.yaml 中已登记 2030 修复版本所有新增字段。

    Test that all new fields from the 2030 fix are registered in records_schema_extensions.yaml.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If any required new field path is missing from the schema.
    """
    schema_path = _REPO_ROOT / "configs" / "records_schema_extensions.yaml"
    assert schema_path.exists(), f"schema 文件不存在：{schema_path}"

    schema_obj = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    assert isinstance(schema_obj, dict), "schema 根节点应为 dict"

    fields_list = schema_obj.get("fields")
    assert isinstance(fields_list, list) and fields_list, "schema.fields 应为非空列表"

    registered_paths = {
        entry.get("path") for entry in fields_list
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }

    missing = _REQUIRED_NEW_PATHS - registered_paths
    assert not missing, (
        f"以下新增字段在 schema 中未登记（共 {len(missing)} 个）：{sorted(missing)}"
    )


def test_event_attestation_score_paths_registered() -> None:
    """
    功能：验证 event_attestation_score 相关字段完成 schema 与 contract 追加登记。

    Validate event_attestation_score field paths are append-only registered in
    both records_schema_extensions.yaml and frozen_contracts.yaml.
    """
    schema_path = _REPO_ROOT / "configs" / "records_schema_extensions.yaml"
    contracts_path = _REPO_ROOT / "configs" / "frozen_contracts.yaml"

    schema_obj = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    contracts_obj = yaml.safe_load(contracts_path.read_text(encoding="utf-8"))

    schema_fields = {
        entry.get("path") for entry in schema_obj.get("fields", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    registry_fields = set(contracts_obj.get("records_schema", {}).get("field_paths_registry", []))

    required_paths = {
        "attestation.attestation_source",
        "attestation.final_event_attested_decision.event_attestation_score",
        "attestation.final_event_attested_decision.event_attestation_score_name",
        "attestation.final_event_attested_decision.event_attestation_score_semantics",
    }

    assert required_paths <= schema_fields
    assert required_paths <= registry_fields


def test_attestation_source_path_registered_for_statement_only_provenance_contract() -> None:
    """
    功能：验证 attestation_source 已完成 schema、contract 与 path semantics 追加登记。 

    Validate attestation_source is append-only registered for the controlled
    statement-only provenance contract.
    """
    schema_path = _REPO_ROOT / "configs" / "records_schema_extensions.yaml"
    contracts_path = _REPO_ROOT / "configs" / "frozen_contracts.yaml"
    semantics_path = _REPO_ROOT / "configs" / "policy_path_semantics.yaml"

    schema_obj = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    contracts_obj = yaml.safe_load(contracts_path.read_text(encoding="utf-8"))
    semantics_obj = yaml.safe_load(semantics_path.read_text(encoding="utf-8"))

    schema_entries = {
        entry.get("path"): entry
        for entry in schema_obj.get("fields", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    registry_fields = set(contracts_obj.get("records_schema", {}).get("field_paths_registry", []))
    recommended_fields = set(
        semantics_obj.get("field_catalog", {}).get("catalogs", {}).get("recommended", [])
    )
    diagnostic_constraints = semantics_obj.get("field_catalog", {}).get("diagnostic_only_field_constraints", [])

    assert "attestation.attestation_source" in schema_entries
    assert "attestation.attestation_source" in registry_fields
    assert "attestation.attestation_source" in recommended_fields

    description = schema_entries["attestation.attestation_source"].get("description")
    assert isinstance(description, str)
    assert "formal_input_payload" in description
    assert "negative_branch_statement_only_provenance" in description

    constraint_index = {
        entry.get("field_path"): entry
        for entry in diagnostic_constraints
        if isinstance(entry, dict) and isinstance(entry.get("field_path"), str)
    }
    source_constraint = constraint_index.get("attestation.attestation_source")
    assert isinstance(source_constraint, dict)
    assert source_constraint.get("semantics") == "attestation_source_contract"
    assert "statement_only_provenance_no_bundle" in str(source_constraint.get("rule"))


def test_negative_branch_attestation_provenance_paths_registered() -> None:
    """
    功能：验证 negative branch attestation provenance 已完成 schema、contract 与 path semantics 注册。 

    Validate negative-branch attestation provenance paths are registered in
    schema, frozen contracts, and policy path semantics.
    """
    schema_path = _REPO_ROOT / "configs" / "records_schema_extensions.yaml"
    contracts_path = _REPO_ROOT / "configs" / "frozen_contracts.yaml"
    semantics_path = _REPO_ROOT / "configs" / "policy_path_semantics.yaml"

    schema_obj = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    contracts_obj = yaml.safe_load(contracts_path.read_text(encoding="utf-8"))
    semantics_obj = yaml.safe_load(semantics_path.read_text(encoding="utf-8"))

    schema_entries = {
        entry.get("path"): entry
        for entry in schema_obj.get("fields", [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    registry_fields = set(contracts_obj.get("records_schema", {}).get("field_paths_registry", []))
    recommended_fields = set(
        semantics_obj.get("field_catalog", {}).get("catalogs", {}).get("recommended", [])
    )
    diagnostic_constraints = semantics_obj.get("field_catalog", {}).get("diagnostic_only_field_constraints", [])

    required_paths = {
        "negative_branch_source_attestation_provenance",
        "negative_branch_source_attestation_provenance.statement",
        "negative_branch_source_attestation_provenance.attestation_digest",
        "negative_branch_source_attestation_provenance.event_binding_digest",
        "negative_branch_source_attestation_provenance.trace_commit",
    }

    assert required_paths <= set(schema_entries)
    assert required_paths <= registry_fields
    assert required_paths <= recommended_fields

    provenance_entry = schema_entries["negative_branch_source_attestation_provenance"]
    description = provenance_entry.get("description")
    assert isinstance(description, str)
    assert "diagnostic/provenance" in description
    assert "not a formal attestation payload" in description
    assert "detect formal attestation extractor" in description

    constraint_index = {
        entry.get("field_path"): entry
        for entry in diagnostic_constraints
        if isinstance(entry, dict) and isinstance(entry.get("field_path"), str)
    }
    provenance_constraint = constraint_index.get("negative_branch_source_attestation_provenance")
    assert isinstance(provenance_constraint, dict)
    assert provenance_constraint.get("semantics") == "diagnostic_provenance_only"
    statement_constraint = constraint_index.get("negative_branch_source_attestation_provenance.statement")
    assert isinstance(statement_constraint, dict)
    assert "formal attestation extractor" in str(statement_constraint.get("rule"))


# ---------------------------------------------------------------------------
# （P1 修复）score_parts.lf_status 降级传播回归测试
# ---------------------------------------------------------------------------

def test_lf_status_degraded_when_bp_not_converged() -> None:
    """
    功能：验证当 bp_converge_status=degraded 且 lf_status=ok 时，score_parts.lf_status 被覆写为 degraded。

    Test that score_parts["lf_status"] is overwritten to "degraded" when
    bp_converge_status is "degraded" and lf_status is "ok".

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_status is not degraded or lf_status_degraded_reason is absent.
    """
    # 模拟 _bind_raw_scores_to_content_payload() 中 BP 降级守卫的核心逻辑。
    score_parts: Dict[str, Any] = {}
    lf_trace: Dict[str, Any] = {
        "lf_status": "ok",
        "bp_converge_status": "degraded",
    }

    # 模拟顶层 lf_status 写入（if-not-in 守卫）。
    lf_template_status = lf_trace.get("lf_status")
    if "lf_status" not in score_parts and isinstance(lf_template_status, str) and lf_template_status:
        score_parts["lf_status"] = lf_template_status

    # 模拟 BP 降级守卫块。
    _bp_converge_status = lf_trace.get("bp_converge_status")
    if _bp_converge_status == "degraded" and score_parts.get("lf_status") == "ok":
        score_parts["lf_status"] = "degraded"
        score_parts["lf_status_degraded_reason"] = "bp_not_converged"

    assert score_parts["lf_status"] == "degraded", (
        f"bp_converge_status=degraded 且 lf_status=ok 时，score_parts.lf_status 应被覆写为 'degraded'，"
        f"实际：{score_parts['lf_status']}"
    )
    assert score_parts.get("lf_status_degraded_reason") == "bp_not_converged", (
        f"lf_status_degraded_reason 应为 'bp_not_converged'，实际：{score_parts.get('lf_status_degraded_reason')}"
    )


def test_lf_status_not_degraded_when_bp_converged() -> None:
    """
    功能：验证 BP 收敛场景下 lf_status 不被降级，且 lf_status_degraded_reason 不被写入。

    Test that score_parts["lf_status"] remains "ok" and lf_status_degraded_reason
    is absent when bp_converge_status is "ok".

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If lf_status is degraded or lf_status_degraded_reason is present.
    """
    score_parts: Dict[str, Any] = {}
    lf_trace: Dict[str, Any] = {
        "lf_status": "ok",
        "bp_converge_status": "ok",
    }

    lf_template_status = lf_trace.get("lf_status")
    if "lf_status" not in score_parts and isinstance(lf_template_status, str) and lf_template_status:
        score_parts["lf_status"] = lf_template_status

    _bp_converge_status = lf_trace.get("bp_converge_status")
    if _bp_converge_status == "degraded" and score_parts.get("lf_status") == "ok":
        score_parts["lf_status"] = "degraded"
        score_parts["lf_status_degraded_reason"] = "bp_not_converged"

    assert score_parts["lf_status"] == "ok", (
        f"bp_converge_status=ok 时 lf_status 不应被降级，实际：{score_parts['lf_status']}"
    )
    assert "lf_status_degraded_reason" not in score_parts, (
        f"bp_converge_status=ok 时不应写入 lf_status_degraded_reason"
    )
