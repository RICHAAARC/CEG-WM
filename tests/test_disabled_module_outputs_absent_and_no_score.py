"""
Test: Disabled modules output status="absent" and score=None.

功能说明：
- 验证按 ablation 禁用的模块返回 status="absent"、score=None。
- 确保禁用模块不产生失败原因（content_failure_reason / geo_failure_reason 为 None）。
- 验证 absent 语义与配置开关的一致性（enable_content=false → content_result.status="absent"）。

Module type: General module
"""

from typing import Any, Dict
from unittest.mock import MagicMock

from main.watermarking.embed.orchestrator import run_embed_orchestrator
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.core import digests
from main.registries.runtime_resolver import BuiltImplSet


def test_disabled_content_module_outputs_absent():
    """
    功能：验证 ablation 禁用 content 模块时返回 absent 语义。

    Test disabled content module outputs status="absent" and score=None.

    Verifies:
        1. ablation.normalized.enable_content=false → content_result.status="absent".
        2. content_result.score is None.
        3. content_failure_reason is None (absent 状态非错误).
        4. audit.trace_digest is reproducible.

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If disabled module does not output absent correctly.
    """
    # 构造禁用 content 的配置。
    cfg = {
        "policy_path": "content_only",
        "ablation": {
            "normalized": {
                "enable_content": False,  # 禁用 content 链
                "enable_geometry": True,
                "enable_fusion": True,
                "enable_mask": True,
                "enable_subspace": True,
                "enable_rescue": False,
                "enable_lf": True,
                "enable_hf": False,
                "lf_only": False,
                "hf_only": False,
            }
        }
    }

    # 构造mock impl_set（embed orchestrator 仅需 content_extractor / subspace_planner）。
    impl_set = MagicMock(spec=BuiltImplSet)
    impl_set.content_extractor = MagicMock()
    impl_set.subspace_planner = MagicMock()
    # subspace_planner 返回空结果（ablation 不测试 subspace）。
    impl_set.subspace_planner.plan.return_value = MagicMock(
        plan_digest="mock_plan_digest",
        basis_digest="mock_basis_digest",
        plan_stats={},
        plan={}
    )
    impl_set.sync_module = MagicMock()
    impl_set.sync_module.sync.return_value = {}

    cfg_digest = digests.canonical_sha256(cfg)

    # 执行 embed orchestrator（应当旁路 content_extractor.extract，直接返回 absent）。
    record = run_embed_orchestrator(
        cfg=cfg,
        impl_set=impl_set,
        cfg_digest=cfg_digest,
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=None,
        subspace_result_override=None
    )

    # 断言：content_evidence 中 status="absent"、score=None。
    content_evidence = record.get("content_evidence")
    assert content_evidence is not None, "content_evidence must be present in record"
    assert isinstance(content_evidence, dict), "content_evidence must be dict"
    assert content_evidence.get("status") == "absent", "Disabled content module must output status='absent'"
    assert content_evidence.get("score") is None, "Disabled content module must output score=None"
    assert content_evidence.get("content_failure_reason") is None, "Absent status must not have failure reason"

    # 断言：audit 字段存在且 trace_digest 可复算。
    audit = content_evidence.get("audit")
    assert audit is not None, "audit must be present in absent content_evidence"
    assert audit.get("impl_identity") == "ablation_switchboard", "Ablation absent must identify switchboard impl"
    assert isinstance(audit.get("trace_digest"), str), "trace_digest must be non-empty str"

    # 断言：content_extractor.extract 未被调用（因为 enable_content=false 直接短路）。
    impl_set.content_extractor.extract.assert_not_called()


def test_disabled_geometry_module_outputs_absent():
    """
    功能：验证 ablation 禁用 geometry 模块时返回 absent 语义。

    Test disabled geometry module outputs status="absent" and score=None.

    Verifies:
        1. ablation.normalized.enable_geometry=false → geometry_result.status="absent".
        2. geometry_result.geo_score is None.
        3. geo_failure_reason is None (absent 状态非错误).
        4. geometry_extractor.extract 未被调用。

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If disabled module does not output absent correctly.
    """
    # 构造禁用 geometry 的配置。
    cfg = {
        "policy_path": "content_only",
        "target_fpr": 0.01,  # fusion 模块需要 target_fpr。
        "ablation": {
            "normalized": {
                "enable_content": True,
                "enable_geometry": False,  # 禁用 geometry 链
                "enable_fusion": True,
                "enable_mask": True,
                "enable_subspace": True,
                "enable_rescue": False,
                "enable_lf": True,
                "enable_hf": False,
                "lf_only": False,
                "hf_only": False,
            }
        }
    }

    # 构造 mock impl_set（detect orchestrator 需 content_extractor + geometry_extractor + fusion_rule）。
    impl_set = MagicMock(spec=BuiltImplSet)
    impl_set.content_extractor = MagicMock()
    impl_set.geometry_extractor = MagicMock()
    impl_set.subspace_planner = MagicMock()
    impl_set.fusion_rule = MagicMock()

    # content_extractor 返回 ok 结果（测试 geometry 禁用）。
    impl_set.content_extractor.extract.return_value = {
        "status": "ok",
        "score": None,
        "audit": {"impl_identity": "mock_content_extractor", "impl_version": "v1", "impl_digest": "mock_digest", "trace_digest": "mock_trace"},
        "mask_digest": None,
        "mask_stats": None,
        "plan_digest": None,
        "basis_digest": None,
        "lf_trace_digest": None,
        "hf_trace_digest": None,
        "lf_score": None,
        "hf_score": None,
        "score_parts": {},
        "content_failure_reason": None
    }

    # subspace_planner 返回空结果。
    impl_set.subspace_planner.plan.return_value = MagicMock(
        plan_digest="mock_plan_digest",
        basis_digest="mock_basis_digest",
        plan={}
    )

    # fusion_rule 返回 ok 决策。
    impl_set.fusion_rule.fuse.return_value = MagicMock(
        is_watermarked=False,
        decision_status="ok",
        abstain_reason=None,
        fusion_rule_version="v1"
    )

    cfg_digest = digests.canonical_sha256(cfg)

    # 执行 detect orchestrator（应当旁路 geometry_extractor.extract，直接返回 absent）。
    record = run_detect_orchestrator(
        cfg=cfg,
        impl_set=impl_set,
        input_record=None,
        cfg_digest=cfg_digest,
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=None,
        detect_plan_result_override=None
    )

    # 断言：geometry_evidence_payload 中 status="absent"、geo_score=None。
    geometry_evidence = record.get("geometry_evidence_payload")
    assert geometry_evidence is not None, "geometry_evidence_payload must be present in record"
    assert isinstance(geometry_evidence, dict), "geometry_evidence must be dict"
    assert geometry_evidence.get("status") == "absent", "Disabled geometry module must output status='absent'"
    assert geometry_evidence.get("geo_score") is None, "Disabled geometry module must output geo_score=None"
    assert geometry_evidence.get("geo_failure_reason") is None, "Absent status must not have failure reason"

    # 断言：audit 字段存在且 impl_identity 为 ablation_switchboard。
    audit = geometry_evidence.get("audit")
    assert audit is not None, "audit must be present in absent geometry_evidence"
    assert audit.get("impl_identity") == "ablation_switchboard", "Ablation absent must identify switchboard impl"
    assert isinstance(audit.get("trace_digest"), str), "trace_digest must be non-empty str"

    # 断言：geometry_extractor.extract 未被调用（因为 enable_geometry=false 直接短路）。
    impl_set.geometry_extractor.extract.assert_not_called()


def test_ablation_absent_reason_is_traceable():
    """
    功能：验证 ablation absent 原因可通过 trace_digest 追溯。

    Test ablation absent reason is traceable via trace_digest.

    Verifies:
        1. Different absent reasons yield different trace_digest values.
        2. Same absent reason yields same trace_digest (reproducible).

    Returns:
        None (asserts on failure).

    Raises:
        AssertionError: If trace_digest is not traceable or reproducible.
    """
    from main.watermarking.embed.orchestrator import _build_ablation_absent_content_evidence
    from main.watermarking.detect.orchestrator import _build_ablation_absent_geometry_evidence

    # 构造两种不同的 absent reason。
    content_absent = _build_ablation_absent_content_evidence("content_chain_disabled_by_ablation")
    geometry_absent = _build_ablation_absent_geometry_evidence("geometry_chain_disabled_by_ablation")

    # 断言：不同原因的 trace_digest 必须不同。
    content_trace = content_absent["audit"]["trace_digest"]
    geometry_trace = geometry_absent["audit"]["trace_digest"]
    assert content_trace != geometry_trace, "Different absent reasons must yield different trace_digest"

    # 断言：相同原因的 trace_digest 必须可复现。
    content_absent_2 = _build_ablation_absent_content_evidence("content_chain_disabled_by_ablation")
    assert content_absent["audit"]["trace_digest"] == content_absent_2["audit"]["trace_digest"], "trace_digest must be reproducible"


def test_embed_execution_chain_status_normalizes_deprecated_fail() -> None:
    """
    功能：验证 embed 链路状态归一化会将 fail 统一映射为 failed。

    Verify embed execution-chain status normalization maps deprecated fail to failed.

    Returns:
        None.

    Raises:
        AssertionError: If deprecated fail is not normalized to failed.
    """
    from main.watermarking.embed.orchestrator import _normalize_execution_chain_status

    assert _normalize_execution_chain_status("fail") == "failed"
    assert _normalize_execution_chain_status("ok") == "ok"
    assert _normalize_execution_chain_status("absent") == "absent"
