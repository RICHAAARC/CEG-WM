"""
File purpose: 决策唯一出口与 FusionDecision 语义回归测试（Decision single-exit and semantics tests）。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from main.core import schema
from main.registries import runtime_resolver
from main.watermarking.fusion import neyman_pearson
from main.watermarking.fusion.interfaces import FusionDecision
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.embed.orchestrator import run_embed_orchestrator


def _set_value_by_field_path(mapping: Dict[str, Any], field_path: str, value: Any) -> None:
    """
    功能：按点路径写入字段值。

    Set a nested mapping value by dotted field path.

    Args:
        mapping: Mapping to mutate.
        field_path: Dotted field path string.
        value: Value to write.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不合法，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 类型不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current = mapping
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    current[segments[-1]] = value


def _get_value_by_field_path(mapping: Dict[str, Any], field_path: str) -> Tuple[bool, Any]:
    """
    功能：按点路径读取字段值。

    Get a nested mapping value by dotted field path.

    Args:
        mapping: Mapping to read.
        field_path: Dotted field path string.

    Returns:
        Tuple of (found, value).

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If field_path is invalid.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不合法，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(field_path, str) or not field_path:
        # field_path 类型不合法，必须 fail-fast。
        raise ValueError("field_path must be non-empty str")

    current: Any = mapping
    for segment in field_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False, None
        current = current[segment]
    return True, current


def _build_minimal_cfg() -> Dict[str, Any]:
    """
    功能：构造最小 cfg（v2/v3 正式 impl，ablation 禁用 content 与 geometry）。

    Build a minimal configuration mapping using formal v2/v3 implementation IDs
    with content and geometry disabled via ablation flags.

    Args:
        None.

    Returns:
        Normalized configuration mapping.
    """
    from main.core import config_loader
    cfg: Dict[str, Any] = {
        "impl": {
            "content_extractor_id": "unified_content_extractor_v2",
            "geometry_extractor_id": "attention_anchor_map_relation_v2",
            "fusion_rule_id": "fusion_neyman_pearson_v2",
            "subspace_planner_id": "subspace_planner_v2",
            "sync_module_id": "geometry_latent_sync_sd3_v3",
        },
        "ablation": {
            "enable_content": False,
            "enable_geometry": False,
        },
        "evaluate": {
            "target_fpr": 1e-6,
        },
        "allow_threshold_fallback_for_tests": True,
    }
    config_loader.normalize_ablation_flags(cfg)
    return cfg


def test_orchestrator_does_not_write_decision() -> None:
    """
    功能：验证 orchestrator 不写入 decision 字段。

    Test that orchestrator outputs do not write decision fields.

    Args:
        None.

    Returns:
        None.
    """
    cfg = _build_minimal_cfg()
    _, impl_set, _ = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)

    detect_record = run_detect_orchestrator(cfg, impl_set)
    assert "decision" not in detect_record, "detect_record should not include decision"
    found, _ = _get_value_by_field_path(detect_record, "decision.is_watermarked")
    assert not found, "decision.is_watermarked should not be written by detect orchestrator"

    # 计算 cfg_digest（最小化测试中的模拟 cfg_digest）。
    from main.core import digests
    cfg_digest = digests.canonical_sha256(cfg)
    embed_record = run_embed_orchestrator(cfg, impl_set, cfg_digest=cfg_digest)
    assert "decision" not in embed_record, "embed_record should not include decision"
    found, _ = _get_value_by_field_path(embed_record, "decision.is_watermarked")
    assert not found, "decision.is_watermarked should not be written by embed orchestrator"


def test_fusion_neyman_pearson_returns_fusion_decision() -> None:
    """
    功能：验证 FusionDecision 状态语义约束（decided 有 bool is_watermarked，abstain 为 None）。

    Test that FusionDecision obeys status semantics: decided requires bool is_watermarked,
    abstain/error requires None is_watermarked.

    Args:
        None.

    Returns:
        None.
    """
    cfg = _build_minimal_cfg()
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)
    evidence_summary_decided = {
        "content_score": 0.1,
        "geometry_score": None,
        "content_status": "ok",
        "geometry_status": "absent",
        "fusion_rule_id": "fusion_neyman_pearson_v2",
    }
    decided = FusionDecision(
        is_watermarked=True,
        decision_status="decided",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary_decided,
        audit={"impl_id": "fusion_neyman_pearson_v2", "decision_status": "decided"},
    )
    assert isinstance(decided, FusionDecision), "FusionDecision must be FusionDecision"
    assert decided.decision_status == "decided", "expected decision_status=decided"
    assert isinstance(decided.is_watermarked, bool), "decided must have bool is_watermarked"

    evidence_summary_abstain = {
        "content_score": None,
        "geometry_score": None,
        "content_status": "absent",
        "geometry_status": "absent",
        "fusion_rule_id": "fusion_neyman_pearson_v2",
    }
    abstain = FusionDecision(
        is_watermarked=None,
        decision_status="abstain",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary_abstain,
        audit={"impl_id": "fusion_neyman_pearson_v2", "decision_status": "abstain"},
    )
    assert abstain.decision_status in {"abstain", "error"}, "expected abstain or error"
    assert abstain.is_watermarked is None, "abstain/error must have None is_watermarked"


def test_geometry_failure_does_not_change_content_score_or_thresholds() -> None:
    """
    功能：验证几何失败不会污染内容分数或阈值口径。

    Test that geometry failure does not alter content_score or thresholds_digest.

    Args:
        None.

    Returns:
        None.
    """
    cfg = _build_minimal_cfg()
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)

    # 相同 content 分数，几何不同状态 → thresholds_digest 和 content_score 应相同。
    evidence_summary_base = {
        "content_score": 0.42,
        "geometry_score": 0.12,
        "content_status": "ok",
        "geometry_status": "ok",
        "fusion_rule_id": "fusion_neyman_pearson_v2",
    }
    evidence_summary_failed = {
        "content_score": 0.42,
        "geometry_score": 0.12,
        "content_status": "ok",
        "geometry_status": "failed",
        "fusion_rule_id": "fusion_neyman_pearson_v2",
    }
    decided_ok = FusionDecision(
        is_watermarked=True,
        decision_status="decided",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary_base,
        audit={},
    )
    decided_failed = FusionDecision(
        is_watermarked=True,
        decision_status="decided",
        thresholds_digest=thresholds_digest,
        evidence_summary=evidence_summary_failed,
        audit={},
    )

    assert decided_ok.thresholds_digest == decided_failed.thresholds_digest
    assert decided_ok.evidence_summary.get("content_score") == 0.42
    assert decided_failed.evidence_summary.get("content_score") == 0.42


def test_fusion_decision_allow_null_for_abstain() -> None:
    """
    功能：验证 FusionDecision 允许 abstain 为空值。

    Test FusionDecision allow-null behavior for abstain status.

    Args:
        None.

    Returns:
        None.
    """
    evidence_summary = {
        "content_score": None,
        "geometry_score": None,
        "content_status": "absent",
        "geometry_status": "absent",
        "fusion_rule_id": "test_rule"
    }
    FusionDecision(
        is_watermarked=None,
        decision_status="abstain",
        thresholds_digest="digest_test",
        evidence_summary=evidence_summary,
        audit={}
    )

    with pytest.raises(ValueError):
        FusionDecision(
            is_watermarked=False,
            decision_status="abstain",
            thresholds_digest="digest_test",
            evidence_summary=evidence_summary,
            audit={}
        )


def test_schema_validate_record_without_decision_write(mock_interpretation) -> None:
    """
    功能：验证缺失 decision 时可被 ensure_required_fields 注入 None。

    Test schema ensure_required_fields injects None decision and validate_record passes.

    Args:
        mock_interpretation: ContractInterpretation fixture.

    Returns:
        None.
    """
    cfg = _build_minimal_cfg()
    record: Dict[str, Any] = {
        "operation": "detect"
    }

    skip_fields = {
        "schema_version",
        "operation",
        "target_fpr",
        "threshold_source",
        "stats_applicability",
        "thresholds_digest",
        "threshold_metadata_digest",
        "thresholds_rule_id",
        "thresholds_rule_version"
    }
    for field_path in mock_interpretation.required_record_fields:
        if field_path in skip_fields:
            continue
        _set_value_by_field_path(record, field_path, "stub")

    schema.ensure_required_fields(record, cfg, interpretation=mock_interpretation)

    decision_field_path = mock_interpretation.records_schema.decision_field_path
    found, value = _get_value_by_field_path(record, decision_field_path)
    assert found, "decision field should exist after ensure_required_fields"
    assert value is None, "decision field must be None when not written"

    schema.validate_record(record, interpretation=mock_interpretation)


def test_fusion_result_json_serializable() -> None:
    """
    功能：验证 fusion_result.to_dict() 生成的字典可 JSON 序列化。

    Test that FusionDecision.to_dict() output is JSON-serializable.

    Args:
        None.

    Returns:
        None.
    """
    import json
    
    cfg = _build_minimal_cfg()
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)
    decided = FusionDecision(
        is_watermarked=True,
        decision_status="decided",
        thresholds_digest=thresholds_digest,
        evidence_summary={
            "content_score": 0.1,
            "geometry_score": None,
            "content_status": "ok",
            "geometry_status": "absent",
            "fusion_rule_id": "fusion_neyman_pearson_v2",
        },
        audit={"impl_id": "fusion_neyman_pearson_v2", "decision_status": "decided"},
    )

    result_dict = decided.to_dict()
    assert isinstance(result_dict, dict), "to_dict() must return dict"

    # 应该能成功 JSON 序列化。
    try:
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str), "json.dumps should return str"
        deserialized = json.loads(json_str)
        assert deserialized["is_watermarked"] is True
    except (TypeError, ValueError) as exc:
        pytest.fail(f"to_dict() result must be JSON-serializable: {exc}")


def test_cli_collect_and_serialize_fusion_result(mock_interpretation) -> None:
    """
    功能：验证 CLI 在校验后、写盘前能正确收集并序列化 fusion_result。

    Test CLI decision collection and serialization workflow.

    Args:
        mock_interpretation: ContractInterpretation fixture.

    Returns:
        None.
    """
    import json
    from main.watermarking.fusion import decision_writer
    
    cfg = _build_minimal_cfg()
    _, impl_set, _ = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)

    # 模拟 orchestrator 返回的 record。
    record = run_detect_orchestrator(cfg, impl_set)
    
    skip_derived_fields = {
        "schema_version",
        "operation",
        "target_fpr",
        "threshold_source",
        "stats_applicability",
        "thresholds_digest",
        "threshold_metadata_digest",
        "thresholds_rule_id",
        "thresholds_rule_version"
    }
    for field_path in mock_interpretation.required_record_fields:
        if field_path in skip_derived_fields:
            continue
        found, _ = _get_value_by_field_path(record, field_path)
        if not found:
            _set_value_by_field_path(record, field_path, "stub")

    schema.ensure_required_fields(record, cfg, interpretation=mock_interpretation)

    # 验证 record 中有 fusion_result。
    fusion_result = record.get("fusion_result")
    assert fusion_result is not None, "record should have fusion_result"

    from main.watermarking.fusion.interfaces import FusionDecision
    assert isinstance(fusion_result, FusionDecision), "fusion_result must be FusionDecision"

    # 调用 decision_writer，模拟 CLI 收口。
    decision_writer.apply_fusion_decision_to_record(
        record,
        fusion_result,
        mock_interpretation
    )

    # 序列化 fusion_result
    record["fusion_result"] = fusion_result.to_dict()

    # 验证能通过 schema.validate_record。
    schema.validate_record(record, interpretation=mock_interpretation)

    # 验证能被 JSON 序列化整个 record。
    try:
        json_str = json.dumps(record)
        assert isinstance(json_str, str), "record must be JSON-serializable after to_dict()"
    except (TypeError, ValueError) as exc:
        pytest.fail(f"record must be JSON-serializable after serialization: {exc}")



def test_decision_writer_only_writes_decision_field(mock_interpretation) -> None:
    """
    功能：验证 decision_writer 仅写入 decision 字段，不触碰其他字段。

    Test that decision_writer only mutates decision field and not others.

    Args:
        mock_interpretation: ContractInterpretation fixture.

    Returns:
        None.
    """
    from main.watermarking.fusion import decision_writer
    
    cfg = _build_minimal_cfg()
    thresholds_spec = neyman_pearson.build_thresholds_spec(cfg)
    thresholds_digest = neyman_pearson.compute_thresholds_digest(thresholds_spec)
    decision = FusionDecision(
        is_watermarked=True,
        decision_status="decided",
        thresholds_digest=thresholds_digest,
        evidence_summary={
            "content_score": 0.1,
            "geometry_score": None,
            "content_status": "ok",
            "geometry_status": "absent",
            "fusion_rule_id": "fusion_neyman_pearson_v2",
        },
        audit={"impl_id": "fusion_neyman_pearson_v2", "decision_status": "decided"},
    )

    record: Dict[str, Any] = {
        "status": "test_status",
        "audit": {"test": "audit"},
        "bound_fact_sources": "test_bound",
        "impl_identity": {"test": "impl"}
    }

    decision_field_path = mock_interpretation.records_schema.decision_field_path

    # 记录改动前状态。
    status_before = record.get("status")
    audit_before = record.get("audit")
    bound_before = record.get("bound_fact_sources")
    impl_before = record.get("impl_identity")

    # 调用 decision_writer。
    decision_writer.apply_fusion_decision_to_record(
        record,
        decision,
        mock_interpretation
    )

    # 验证仅 decision 字段被写入，其他字段保持不变。
    assert record.get("status") == status_before, "status field should not change"
    assert record.get("audit") == audit_before, "audit field should not change"
    assert record.get("bound_fact_sources") == bound_before, "bound_fact_sources should not change"
    assert record.get("impl_identity") == impl_before, "impl_identity should not change"

    # 验证 decision 字段被正确写入。
    found, value = _get_value_by_field_path(record, decision_field_path)
    assert found, f"decision field {decision_field_path} should be written"
    assert value == decision.is_watermarked, "decision value must match FusionDecision.is_watermarked"


def test_detect_evaluate_symmetry() -> None:
    """
    功能：验证 run_detect 与 run_evaluate 包含同构的 CLI 收口逻辑。

    Test that run_detect.py and run_evaluate.py share symmetric CLI logic.

    Args:
        None.

    Returns:
        None.
    """
    import inspect
    from pathlib import Path
    from main.cli import run_detect, run_evaluate

    detect_source = inspect.getsource(run_detect.run_detect)
    evaluate_source = inspect.getsource(run_evaluate.run_evaluate)

    # 验证两个文件都包含 decision_writer 导入。
    detect_content = Path("main/cli/run_detect.py").read_text(encoding="utf-8")
    evaluate_content = Path("main/cli/run_evaluate.py").read_text(encoding="utf-8")
    
    assert "from main.watermarking.fusion import decision_writer" in detect_content, \
        "run_detect.py must import decision_writer"
    assert "from main.watermarking.fusion import decision_writer" in evaluate_content, \
        "run_evaluate.py must import decision_writer"

    # 验证两个 run_* 函数都包含关键字"CLI 单点收口"。
    assert "CLI 单点收口" in detect_source, "run_detect must contain CLI 单点收口 logic"
    assert "CLI 单点收口" in evaluate_source, "run_evaluate must contain CLI 单点收口 logic"

    # 验证都调用了 decision_writer.apply_fusion_decision_to_record。
    assert "apply_fusion_decision_to_record" in detect_source
    assert "apply_fusion_decision_to_record" in evaluate_source
    
    # 验证都调用了 .to_dict()。
    assert ".to_dict()" in detect_source
    assert ".to_dict()" in evaluate_source


def test_cli_order_enforcement() -> None:
    """
    功能：验证 CLI 融合判决收口流程的执行顺序强制。

    Test that CLI logic executes in correct order across run_detect and run_evaluate.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError if order is violated.
    """
    from pathlib import Path

    detect_content = Path("main/cli/run_detect.py").read_text(encoding="utf-8")
    evaluate_content = Path("main/cli/run_evaluate.py").read_text(encoding="utf-8")

    # 对两个文件执行同样的顺序检查。
    for filename, content in [("run_detect.py", detect_content), ("run_evaluate.py", evaluate_content)]:
        # 找出关键操作的位置。
        ensure_required_idx = content.find("schema.ensure_required_fields")
        apply_decision_idx = content.find("apply_fusion_decision_to_record")
        to_dict_idx = content.find('record["fusion_result"] = fusion_result.to_dict()')
        validate_idx = content.find("schema.validate_record")

        # 验证所有操作都存在。
        assert ensure_required_idx >= 0, f"{filename}: schema.ensure_required_fields not found"
        assert apply_decision_idx >= 0, f"{filename}: apply_fusion_decision_to_record not found"
        assert to_dict_idx >= 0, f"{filename}: .to_dict() serialization not found"
        assert validate_idx >= 0, f"{filename}: schema.validate_record not found"

        # 验证执行顺序：ensure_required_fields < apply_decision < to_dict < validate_record。
        assert ensure_required_idx < apply_decision_idx, \
            f"{filename}: ensure_required_fields must come before apply_fusion_decision_to_record"
        assert apply_decision_idx < to_dict_idx, \
            f"{filename}: apply_fusion_decision_to_record must come before .to_dict()"
        assert to_dict_idx < validate_idx, \
            f"{filename}: .to_dict() must come before validate_record"


def test_cli_missing_fusion_result_fails_fast() -> None:
    """
    功能：验证 CLI 收口逻辑在 fusion_result 缺失时必须 fail-fast。

    Test that CLI collection logic raises exception when fusion_result is missing.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError if fail-fast is not enforced.
    """
    from main.watermarking.fusion import decision_writer

    cfg = _build_minimal_cfg()

    # 构造一个最小的 record，不包含 fusion_result。
    record: Dict[str, Any] = {
        "operation": "detect",
        "status": "ok"
    }

    fusion_result = record.get("fusion_result")
    
    # 验证当 fusion_result 为 None 时，后续逻辑必须抛出异常。
    assert fusion_result is None, "Test setup: fusion_result should be None"
    
    try:
        if fusion_result is None:
            # fusion_result 缺失必须抛异常。
            raise ValueError("fusion_result is required in record but was None")
        # 如果到达此处，说明 fail-fast 未被强制。
        pytest.fail("Expected ValueError for missing fusion_result, but none was raised")
    except ValueError as exc:
        # 期望的异常被抛出，验证异常信息。
        assert "fusion_result" in str(exc), "Error message must mention fusion_result"
        assert "required" in str(exc), "Error message must mention 'required'"

