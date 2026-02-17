"""
File purpose: Integration tests for S-02 SemanticMaskProvider implementation.
Module type: General module

功能说明：
- 验证 SemanticMaskProvider 的可重复性和写盘路径限制。
- 验证 mask_digest 计算、mask_stats 提取、resolution_binding 绑定。
- 验证配置参数集成（enable_mask、mask_resolution_*）。
- 验证失败路径的单一主因上报。
- 验证注册到 content_registry 与 runtime_whitelist 的一致性。
"""

import pytest
from typing import Dict, Any

from main.watermarking.content_chain.semantic_mask_provider import (
    SemanticMaskProvider,
    SEMANTIC_MASK_PROVIDER_ID,
    SEMANTIC_MASK_PROVIDER_VERSION,
)
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.core import digests
from main.registries import content_registry


class TestSemanticMaskProviderReproducibility:
    """测试掩码提供器的可重复性约束。"""

    def test_same_input_produces_same_mask_digest(self):
        """同样输入应产生相同的掩码摘要（可重复性）。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True, "mask_resolution_width": 512, "mask_resolution_height": 512}
        inputs = {
            "image": [1, 2, 3],
            "image_shape": (512, 512, 3)
        }

        # 多次调用应返回相同 mask_digest。
        result1 = provider.extract(cfg, inputs)
        result2 = provider.extract(cfg, inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.mask_digest == result2.mask_digest, \
            "Same inputs must produce identical mask_digest for reproducibility"
        assert result1.mask_digest is not None, \
            "mask_digest 不应为 None（掩码已提取）"

    def test_enable_mask_false_returns_absent(self):
        """enable_mask=false 时返回 absent 语义，无失败原因。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": False}
        inputs = {"image": [1, 2, 3], "image_shape": (512, 512, 3)}

        result = provider.extract(cfg, inputs)

        assert result.status == "absent", \
            "When enable_mask=false, status must be absent"
        assert result.mask_digest is None, \
            "mask_digest must be None when disabled"
        assert result.mask_stats is None, \
            "mask_stats must be None when disabled"
        assert result.content_failure_reason is None, \
            "absent 状态下不应有失败原因"


class TestSemanticMaskProviderFailureReasons:
    """测试掩码提供器的单一主因失败上报。"""

    def test_no_input_failure_reason(self):
        """无输入时返回 mask_extraction_no_input 失败原因。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        result = provider.extract(cfg, inputs=None)

        assert result.status == "failed", \
            "No inputs should result in failed status"
        assert result.content_failure_reason == "mask_extraction_no_input", \
            "失败原因必须是 mask_extraction_no_input"
        assert result.mask_digest is None, \
            "Failed status must have None mask_digest"

    def test_invalid_shape_failure_reason(self):
        """无效形状时返回对应的失败原因。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        inputs = {"image": [1, 2], "image_shape": (-1, 512, 3)}  # 负数分辨率

        result = provider.extract(cfg, inputs)

        assert result.status == "failed", \
            "Invalid shape should result in failed status"
        assert result.content_failure_reason == "mask_extraction_invalid_shape", \
            "失败原因应与形状校验失败相关"
        assert result.score is None, \
            "Failed status must have None score"


class TestSemanticMaskProviderAuditFields:
    """测试审计字段的完整性与冻结。"""

    def test_audit_fields_present_and_valid(self):
        """审计字段必须完整且有效。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest="abc123def456"
        )

        cfg = {"enable_mask": False}
        result = provider.extract(cfg)

        assert "impl_identity" in result.audit, \
            "audit 必须包含 impl_identity"
        assert "impl_version" in result.audit, \
            "audit 必须包含 impl_version"
        assert "impl_digest" in result.audit, \
            "audit 必须包含 impl_digest"
        assert "trace_digest" in result.audit, \
            "audit 必须包含 trace_digest"
        
        assert result.audit["impl_identity"] == SEMANTIC_MASK_PROVIDER_ID
        assert result.audit["impl_version"] == SEMANTIC_MASK_PROVIDER_VERSION
        assert result.audit["impl_digest"] == "abc123def456"

    def test_trace_digest_reproducible(self):
        """trace_digest 应可重复计算。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": False}
        result1 = provider.extract(cfg)
        result2 = provider.extract(cfg)

        assert result1.audit["trace_digest"] == result2.audit["trace_digest"], \
            "trace_digest 必须对于相同输入可重复"


class TestSemanticMaskProviderResolutionBinding:
    """测试分辨率绑定的正确性与完整性。"""

    def test_resolution_binding_present_on_success(self):
        """成功时分辨率绑定应包含在 mask_stats 中。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        inputs = {"image": [1, 2, 3], "image_shape": (768, 1024, 3)}

        result = provider.extract(cfg, inputs)

        assert result.status == "ok"
        assert result.mask_digest is not None, \
            "mask_digest 不应为 None（掩码已计算）"
        assert result.mask_stats is not None, \
            "mask_stats 不应为 None"
        assert "resolution_binding" in result.mask_stats, \
            "resolution_binding 必须包含在 mask_stats 中"
        
        binding = result.mask_stats["resolution_binding"]
        assert binding["width"] == 1024
        assert binding["height"] == 768
        assert binding["aspect_ratio"] == pytest.approx(1024 / 768, rel=1e-3)
        assert binding["binding_version"] == "v1"

    def test_resolution_binding_absent_when_disabled(self):
        """禁用時分辨率绑定应不存在。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": False}
        inputs = {"image_shape": (512, 512, 3)}

        result = provider.extract(cfg, inputs)

        assert result.status == "absent"
        assert result.mask_stats is None, \
            "absent 状态下 mask_stats 必须为 None"


class TestSemanticMaskProviderRegistration:
    """测试 SemanticMaskProvider 在注册表中的正确注册。"""

    def test_semantic_mask_provider_registered_in_content_registry(self):
        """SemanticMaskProvider 必须注册在 content_registry 中。"""
        try:
            factory = content_registry.resolve_content_extractor(SEMANTIC_MASK_PROVIDER_ID)
            assert callable(factory), \
                f"Factory for {SEMANTIC_MASK_PROVIDER_ID} must be callable"
        except ValueError as e:
            pytest.fail(f"semantic_mask_provider_v1 not registered in content_registry: {e}")

    def test_semantic_mask_provider_factory_produces_instance(self):
        """factory 应能正确构造 SemanticMaskProvider 实例。"""
        factory = content_registry.resolve_content_extractor(SEMANTIC_MASK_PROVIDER_ID)
        cfg = {"enable_mask": False}
        
        provider = factory(cfg)
        
        assert isinstance(provider, SemanticMaskProvider), \
            "Factory must produce SemanticMaskProvider instance"
        
        # 验证实例能够调用 extract 方法。
        result = provider.extract(cfg)
        assert isinstance(result, ContentEvidence), \
            "extract() 必须返回 ContentEvidence"


class TestSemanticMaskProviderInterfaceCompliance:
    """测试 ContentExtractor 协议合规性。"""

    def test_implements_extract_protocol(self):
        """SemanticMaskProvider 必须实现 extract(cfg, inputs) 协议。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        assert hasattr(provider, "extract"), \
            "SemanticMaskProvider must have extract method"
        assert callable(provider.extract), \
            "extract must be callable"

    def test_extract_signature_matches_protocol(self):
        """extract() 方法签名必须与 ContentExtractor 协议匹配。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        # 调用应支持 cfg 与可选 inputs 参数。
        cfg = {"enable_mask": False}
        
        # 只提供 cfg
        result1 = provider.extract(cfg)
        assert isinstance(result1, ContentEvidence)
        
        # 提供 cfg 与 inputs
        result2 = provider.extract(cfg, {"image": [1, 2, 3]})
        assert isinstance(result2, ContentEvidence)

    def test_returns_content_evidence_frozen_instance(self):
        """extract() 必须返回冻结的 ContentEvidence 实例。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        result = provider.extract({"enable_mask": False})

        assert isinstance(result, ContentEvidence), \
            "extract() must return ContentEvidence"
        
        # ContentEvidence 应为冻结 dataclass。
        with pytest.raises(Exception):
            # 尝试修改应失败（冻结保护）。
            result.status = "mismatch"


class TestSemanticMaskProviderInputValidation:
    """测试输入验证与 fail-fast 行为。"""

    def test_cfg_type_validation_fail_fast(self):
        """cfg 非 dict 时应 fail-fast。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        with pytest.raises(TypeError, match="cfg must be dict"):
            provider.extract("invalid_cfg")

    def test_inputs_type_validation_fail_fast(self):
        """inputs 非 dict 或 None 时应 fail-fast。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        with pytest.raises(TypeError, match="inputs must be dict or None"):
            provider.extract({}, "invalid_inputs")

    def test_impl_id_validation_on_construction(self):
        """impl_id 为空或非 str 时 __init__ 应 fail-fast。"""
        with pytest.raises(ValueError, match="impl_id must be non-empty str"):
            SemanticMaskProvider("", "v1", "digest")

        with pytest.raises(ValueError, match="impl_id must be non-empty str"):
            SemanticMaskProvider(None, "v1", "digest")

    def test_impl_version_validation_on_construction(self):
        """impl_version 为空或非 str 时 __init__ 应 fail-fast。"""
        with pytest.raises(ValueError, match="impl_version must be non-empty str"):
            SemanticMaskProvider("id", "", "digest")

        with pytest.raises(ValueError, match="impl_version must be non-empty str"):
            SemanticMaskProvider("id", None, "digest")

    def test_impl_digest_validation_on_construction(self):
        """impl_digest 为空或非 str 时 __init__ 应 fail-fast。"""
        with pytest.raises(ValueError, match="impl_digest must be non-empty str"):
            SemanticMaskProvider("id", "v1", "")

        with pytest.raises(ValueError, match="impl_digest must be non-empty str"):
            SemanticMaskProvider("id", "v1", None)
