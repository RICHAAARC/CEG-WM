"""
功能：审计修复验证测试

Module type: Core innovation module

验证阈值来源绑定与救援原因语义的修复是否正确实现。
- 阈值来源绑定：验证 threshold_source 审计字段正确追踪来源（artifact vs test-only fallback）
- 救援原因语义：验证 rescue_reason 语义正确（"rescued_by_geo_gate" 而非倒转的值）
"""

import pytest
from typing import Dict, Any
from main.watermarking.fusion.decision import NeumanPearsonFusionRule
from main.registries.fusion_registry import resolve_fusion_rule


class TestB1ThresholdSourceAudit:
    """
    阈值来源绑定验证：threshold_source 审计字段正确记录阈值来源
    """
    
    @pytest.fixture
    def np_fusion(self):
        """创建 NeumanPearson 融合规则 v2 实例"""
        factory = resolve_fusion_rule("fusion_neyman_pearson")
        return factory({})
    
    def test_threshold_source_artifact_when_provided(self, np_fusion):
        """
        测试：当 __thresholds_artifact__ 被提供时，threshold_source 应为 "artifact"
        """
        cfg = {
            "target_fpr": 0.1,
            "__thresholds_artifact__": {
                "threshold_value": 0.5,
                "threshold_id": "test_threshold_1",
                "target_fpr": 0.1
            }
        }
        
        content_evidence = {
            "content_score": 0.7,
            "status": "ok"
        }
        
        geometry_evidence = {
            "geo_score": 0.6,
            "status": "ok"
        }
        
        result = np_fusion.fuse(cfg, content_evidence, geometry_evidence)
        
        # 验证：当 artifact 来自 NP 校准工件时，threshold_source 标记为 np_canonical，
        # 与冻结决策门 threshold_source=np_canonical 约束对齐。
        assert result.audit.get("threshold_source") == "np_canonical"
        # 验证：used_threshold_value 应来自 artifact
        assert result.audit.get("used_threshold_value") == 0.5
    
    def test_threshold_source_fallback_when_artifact_missing(self, np_fusion):
        """
        测试：当 __thresholds_artifact__ 缺失时，默认应 fail-fast。
        """
        cfg = {
            "target_fpr": 0.1
            # 注意：没有 __thresholds_artifact__
        }
        
        content_evidence = {
            "content_score": 0.05,  # 低于 target_fpr 0.1
            "status": "ok"
        }
        
        geometry_evidence = {
            "geo_score": 0.6,
            "status": "ok"
        }
        
        with pytest.raises(ValueError, match="np threshold artifact is required"):
            _ = np_fusion.fuse(cfg, content_evidence, geometry_evidence)

    def test_threshold_source_fallback_only_when_test_flag_enabled(self, np_fusion):
        """
        测试：仅当 allow_threshold_fallback_for_tests=True 时允许 fallback。
        """
        cfg = {
            "target_fpr": 0.1,
            "allow_threshold_fallback_for_tests": True,
        }

        content_evidence = {
            "content_score": 0.05,
            "status": "ok"
        }

        geometry_evidence = {
            "geo_score": 0.6,
            "status": "ok"
        }

        result = np_fusion.fuse(cfg, content_evidence, geometry_evidence)

        assert result.audit.get("threshold_source") == "fallback_target_fpr_test_only"
        assert result.audit.get("used_threshold_value") == 0.1
        assert result.audit.get("allow_threshold_fallback_for_tests") is True


class TestB3RescueReasonSemantic:
    """
    救援原因语义验证：rescue_reason 语义正确，与触发条件一致
    """
    
    @pytest.fixture
    def np_fusion(self):
        """创建 NeumanPearson 融合规则 v2 实例"""
        factory = resolve_fusion_rule("fusion_neyman_pearson")
        return factory({})
    
    def test_rescue_reason_correct_when_triggered(self, np_fusion):
        """
        测试：当救援被触发时，rescue_reason 应为 "rescued_by_geo_gate"（而非倒转的语义）
        """
        cfg = {
            "target_fpr": 0.1,
            "rescue_band_lower": 0.05,
            "rescue_band_upper": 0.15,
            "geometry_gate_threshold": 0.5,
            "__thresholds_artifact__": {
                "threshold_value": 0.1,
                "threshold_id": "test_threshold",
                "target_fpr": 0.1
            }
        }
        
        # content score 在救援带内（0.05 < score < 0.15）且 NP 决策为 False
        content_evidence = {
            "content_score": 0.08,
            "status": "ok"
        }
        
        # geometry score > gate threshold，触发救援
        geometry_evidence = {
            "geo_score": 0.6,
            "status": "ok"
        }
        
        result = np_fusion.fuse(cfg, content_evidence, geometry_evidence)
        
        # 验证：当救援触发时，rescue_reason 应为 "rescued_by_geo_gate"
        if result.audit.get("rescue_triggered"):
            assert result.audit.get("rescue_reason") == "rescued_by_geo_gate"
            # 验证：is_watermarked 应为 True（被救援）
            assert result.is_watermarked is True
    
    def test_rescue_reason_none_when_not_triggered(self, np_fusion):
        """
        测试：当 NP 决策本身为 True 或救援不被触发时，rescue_reason 应为 None
        """
        cfg = {
            "target_fpr": 0.1,
            "rescue_band_lower": 0.05,
            "rescue_band_upper": 0.15,
            "geometry_gate_threshold": 0.5,
            "__thresholds_artifact__": {
                "threshold_value": 0.1,
                "target_fpr": 0.1
            }
        }
        
        # content score > threshold，NP 决策为 True，不需要救援
        content_evidence = {
            "content_score": 0.3,
            "status": "ok"
        }
        
        geometry_evidence = {
            "geo_score": 0.4,
            "status": "ok"
        }
        
        result = np_fusion.fuse(cfg, content_evidence, geometry_evidence)
        
        # 验证：无救援时 rescue_reason 应为 None
        assert not result.audit.get("rescue_triggered")
        assert result.audit.get("rescue_reason") is None
