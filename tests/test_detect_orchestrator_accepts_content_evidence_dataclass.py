"""
File purpose: Regression tests for detect orchestrator with ContentEvidence dataclass.
Module type: General module

功能说明：
- 验证 detect 编排器兼容 ContentEvidence 数据类（不仅仅是 dict）。
- 验证融合规则入口能从 ContentEvidence 适配字典中读取预期的字段。
- 验证 content_detector_v1 实现下的完整流程（extract → adapt → fuse）。
"""

from typing import Any, Dict
from unittest.mock import Mock, MagicMock

import pytest

from main.watermarking.content_chain.content_detector import ContentDetector, CONTENT_DETECTOR_ID, CONTENT_DETECTOR_VERSION
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.watermarking.detect.orchestrator import (
    run_detect_orchestrator,
    _adapt_content_evidence_for_fusion,
    _adapt_geometry_evidence_for_fusion
)
from main.registries.runtime_resolver import BuiltImplSet


class TestContentEvidenceDataclassAdaptation:
    """测试 ContentEvidence 数据类到字典的转换。"""

    def test_adapt_content_evidence_calls_as_dict_method(self) -> None:
        """as_dict() 方法被调用并返回预期字典。"""
        evidence = ContentEvidence(
            status="ok",
            score=0.75,
            audit={"impl_identity": "test_impl", "impl_version": "v1", "impl_digest": "abc123", "trace_digest": "def456"},
            lf_score=0.5,
            hf_score=0.9,
            score_parts={"lf_score": 0.5, "hf_score": 0.9}
        )
        
        adapted = _adapt_content_evidence_for_fusion(evidence)
        
        # 验证关键字段存在。
        assert "status" in adapted
        assert adapted["status"] == "ok"
        assert "score" in adapted
        assert adapted["score"] == 0.75
        assert "audit" in adapted
        assert "lf_score" in adapted
        assert adapted["lf_score"] == 0.5

    def test_adapt_content_evidence_from_dict(self) -> None:
        """直接输入字典时直接返回。"""
        input_dict = {
            "status": "ok",
            "score": 0.8,
            "audit": {"impl_identity": "test", "impl_version": "v1", "impl_digest": "x", "trace_digest": "y"}
        }
        
        adapted = _adapt_content_evidence_for_fusion(input_dict)
        
        assert adapted is input_dict
        assert adapted["status"] == "ok"
        assert adapted["score"] == 0.8

    def test_adapt_content_evidence_absent_status(self) -> None:
        """absent 状态的 ContentEvidence 被正确适配。"""
        evidence = ContentEvidence(
            status="absent",
            score=None,
            audit={"impl_identity": "test", "impl_version": "v1", "impl_digest": "x", "trace_digest": "y"},
            content_failure_reason=None
        )
        
        adapted = _adapt_content_evidence_for_fusion(evidence)
        
        assert adapted["status"] == "absent"
        assert adapted["score"] is None

    def test_adapt_content_evidence_mismatch_status(self) -> None:
        """mismatch 状态的 ContentEvidence 被正确适配。"""
        evidence = ContentEvidence(
            status="mismatch",
            score=None,
            audit={"impl_identity": "detector", "impl_version": "v1", "impl_digest": "xyz", "trace_digest": "abc"},
            plan_digest="digest123",
            content_failure_reason="detector_plan_mismatch"
        )
        
        adapted = _adapt_content_evidence_for_fusion(evidence)
        
        assert adapted["status"] == "mismatch"
        assert adapted["score"] is None
        assert adapted["content_failure_reason"] == "detector_plan_mismatch"


class TestGeometryEvidenceDataclassAdaptation:
    """测试 GeometryEvidence 数据类到字典的转换。"""

    def test_adapt_geometry_evidence_from_dict(self) -> None:
        """几何证据字典直接返回。"""
        input_dict = {
            "status": "absent",
            "geo_score": None,
            "audit": {"impl_identity": "geo_baseline", "impl_version": "v1", "impl_digest": "g1", "trace_digest": "g2"}
        }
        
        adapted = _adapt_geometry_evidence_for_fusion(input_dict)
        
        assert adapted is input_dict
        assert adapted["status"] == "absent"


class TestDetectOrchestratorWithContentDetector:
    """测试 detect 编排器与 ContentDetector 集成。"""

    def test_detect_orchestrator_accepts_content_evidence_dataclass(self) -> None:
        """检测编排器成功处理 ContentEvidence 数据类。"""
        # 构造一个 ContentDetector 直接调用。
        detector = ContentDetector(
            impl_id=CONTENT_DETECTOR_ID,
            impl_version=CONTENT_DETECTOR_VERSION,
            impl_digest="test_digest_detector_v1"
        )
        
        # 准备配置和输入（对齐 injection_scope_manifest.yaml 的键空间）。
        cfg = {
            "detect": {
                "content": {"enabled": True}
            },
            "watermark": {
                "plan_digest": "test_plan_digest_123"
            }
        }
        
        inputs = {
            "lf_score": 0.6,
            "hf_score": 0.7,
            "plan_digest": "test_plan_digest_123"
        }
        
        # 调用 content_detector 的 extract 方法。
        evidence = detector.extract(cfg, inputs=inputs, cfg_digest="cfg_dig_123")
        
        # 验证返回的是 ContentEvidence 数据类。
        assert isinstance(evidence, ContentEvidence)
        assert evidence.status == "ok"
        # S-04 阶段：content_score = lf_score（仅 LF 主通道，HF 未启用）
        assert abs(evidence.score - 0.6) < 1e-10  # = lf_score (LF-only in S-04)
        
        # 验证适配转换不报错。
        adapted = _adapt_content_evidence_for_fusion(evidence)
        
        assert isinstance(adapted, dict)
        assert adapted["status"] == "ok"
        # S-04 阶段：score = lf_score
        assert abs(adapted["score"] - 0.6) < 1e-10

    def test_detect_orchestrator_payload_preservation(self) -> None:
        """检测编排器保留完整的 payload（append-only）。"""
        from main.registries.runtime_resolver import BuiltImplSet
        
        # 创建 mock objects 用于测试。
        mock_content_extractor = Mock()
        mock_geometry_extractor = Mock()
        mock_fusion_rule = Mock()
        mock_subspace_planner = Mock()
        
        # 模拟 content_extractor 返回 ContentEvidence 数据类。
        mock_content_evidence = ContentEvidence(
            status="ok",
            score=0.7,
            audit={
                "impl_identity": CONTENT_DETECTOR_ID,
                "impl_version": CONTENT_DETECTOR_VERSION,
                "impl_digest": "test_detector_digest",
                "trace_digest": "test_trace_digest"
            },
            lf_score=0.6,
            hf_score=0.8,
            score_parts={"lf_score": 0.6, "hf_score": 0.8},
            plan_digest="test_plan_digest"
        )
        mock_content_extractor.extract.return_value = mock_content_evidence
        
        # 模拟 geometry_extractor 返回字典（向后兼容）。
        mock_geometry_extractor.extract.return_value = {
            "status": "absent",
            "audit": {"impl_identity": "geo_baseline", "impl_version": "v1", "impl_digest": "g", "trace_digest": "h"}
        }
        
        # 模拟 fusion_rule 返回 FusionDecision。
        mock_fusion_result = Mock()
        mock_fusion_result.evidence_summary = {"content_score": 0.7}
        mock_fusion_rule.fuse.return_value = mock_fusion_result
        
        # 准备 impl_set（使用 MagicMock 以便设置属性）。
        mock_impl_set = MagicMock(spec=BuiltImplSet)
        mock_impl_set.content_extractor = mock_content_extractor
        mock_impl_set.geometry_extractor = mock_geometry_extractor
        mock_impl_set.fusion_rule = mock_fusion_rule
        mock_impl_set.subspace_planner = mock_subspace_planner
        
        cfg = {"watermark": {"detector": {"enabled": True}, "plan_digest": "test_plan"}}
        
        # 调用 detect 编排器。
        record = run_detect_orchestrator(cfg, mock_impl_set, input_record=None, cfg_digest=None)
        
        # 验证 record 包含 content_evidence_payload（append-only）。
        assert "content_evidence_payload" in record
        assert isinstance(record["content_evidence_payload"], dict)
        assert record["content_evidence_payload"]["status"] == "ok"
        assert record["content_evidence_payload"]["score"] == 0.7
        
        # 验证 geometry_evidence_payload 也存在。
        assert "geometry_evidence_payload" in record
        assert isinstance(record["geometry_evidence_payload"], dict)
        assert record["geometry_evidence_payload"]["status"] == "absent"
        
        # 验证融合规则被调用时接收到的是适配的字典，而不是原始数据类。
        # 检查 fuse 的调用参数：它应该收到适配的内容证据（字典）。
        call_args = mock_fusion_rule.fuse.call_args
        assert call_args is not None
        content_arg = call_args[0][1]  # 第二个位置参数是 content 证据。
        assert isinstance(content_arg, dict), "融合规则应该收到字典，而不是数据类"
        assert content_arg["status"] == "ok"
        assert content_arg["score"] == 0.7

    def test_detect_orchestrator_fusion_receives_adapted_dict(self) -> None:
        """验证融合规则接收到适配的字典。"""
        from main.registries.runtime_resolver import BuiltImplSet
        
        mock_content_extractor = Mock()
        mock_geometry_extractor = Mock()
        mock_fusion_rule = Mock()
        mock_subspace_planner = Mock()
        
        # ContentDetector 返回 ContentEvidence 数据类，包含 plan_digest 不一致。
        mock_content_evidence = ContentEvidence(
            status="mismatch",
            score=None,
            audit={
                "impl_identity": CONTENT_DETECTOR_ID,
                "impl_version": CONTENT_DETECTOR_VERSION,
                "impl_digest": "detector_digest",
                "trace_digest": "trace"
            },
            plan_digest="plan_digest_mismatch_123",
            content_failure_reason="detector_plan_mismatch"
        )
        mock_content_extractor.extract.return_value = mock_content_evidence
        
        # Geometry 返回字典。
        mock_geometry_extractor.extract.return_value = {
            "status": "absent",
            "audit": {"impl_identity": "g", "impl_version": "v1", "impl_digest": "x", "trace_digest": "y"}
        }
        
        mock_fusion_result = Mock()
        mock_fusion_result.evidence_summary = {"content_score": None}
        mock_fusion_rule.fuse.return_value = mock_fusion_result
        
        mock_impl_set = MagicMock(spec=BuiltImplSet)
        mock_impl_set.content_extractor = mock_content_extractor
        mock_impl_set.geometry_extractor = mock_geometry_extractor
        mock_impl_set.fusion_rule = mock_fusion_rule
        mock_impl_set.subspace_planner = mock_subspace_planner
        
        cfg = {"watermark": {"detector": {"enabled": True}}}
        
        record = run_detect_orchestrator(cfg, mock_impl_set)
        
        # 验证融合收到的字段。
        call_args = mock_fusion_rule.fuse.call_args
        content_arg = call_args[0][1]
        
        assert isinstance(content_arg, dict)
        assert content_arg["status"] == "mismatch"
        assert content_arg["score"] is None
        assert "content_failure_reason" in content_arg
        assert content_arg["content_failure_reason"] == "detector_plan_mismatch"

    def test_content_evidence_payload_roundtrip(self) -> None:
        """ContentEvidence 通过 adapt → dict → 读取保持数据一致性。"""
        original_evidence = ContentEvidence(
            status="ok",
            score=0.82,
            audit={
                "impl_identity": "content_detector_v1",
                "impl_version": "v1",
                "impl_digest": "e9f3e4b",
                "trace_digest": "f8a2c1d"
            },
            lf_score=0.75,
            hf_score=0.89,
            score_parts={"lf_score": 0.75, "hf_score": 0.89},
            plan_digest="plan_abc",
            content_failure_reason=None
        )
        
        # 适配为字典。
        adapted = _adapt_content_evidence_for_fusion(original_evidence)
        
        # 验证数据完整性。
        assert adapted["status"] == original_evidence.status
        assert adapted["score"] == original_evidence.score
        assert adapted["lf_score"] == original_evidence.lf_score
        assert adapted["hf_score"] == original_evidence.hf_score
        assert adapted["plan_digest"] == original_evidence.plan_digest
        assert adapted["audit"] == original_evidence.audit
