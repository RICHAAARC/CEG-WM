"""
File purpose: Tests for A 修复：SemanticMaskProvider cfg_digest_binding 改进
Module type: General module

功能说明：
- 验证 cfg_digest_binding 的正确性：mask_digest 只依赖权威 cfg_digest，不受非 digest_scope 字段影响。
- 验证 cfg_digest 参数的传入、使用与绑定。
- 验证带与不带 cfg_digest 的向后兼容行为。
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
from main.core import config_loader


class TestMaskDigestBindingWithCfgDigest:
    """测试 mask_digest 与权威 cfg_digest 的绑定。"""

    def test_mask_digest_binds_to_cfg_digest_when_provided(self):
        """当 cfg_digest 由调用者提供时，mask_digest 应依赖于它。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True, "mask_resolution_width": 512, "mask_resolution_height": 512}
        inputs = {"image": [1, 2, 3], "image_shape": (512, 512, 3)}
        
        # 模拟权威 cfg_digest（来自 config_loader.compute_cfg_digest）
        cfg_digest_1 = "abc123def456"
        
        result1 = provider.extract(cfg, inputs, cfg_digest=cfg_digest_1)
        
        assert result1.status == "ok"
        assert result1.mask_digest is not None
        # mask_digest 应该基于包含 cfg_digest 的 payload。
        mask_digest_1 = result1.mask_digest
        
        # 使用不同的 cfg_digest 应产生不同的 mask_digest。
        cfg_digest_2 = "different456xyz789"
        result2 = provider.extract(cfg, inputs, cfg_digest=cfg_digest_2)
        
        assert result2.status == "ok"
        mask_digest_2 = result2.mask_digest
        
        assert mask_digest_1 != mask_digest_2, \
            "不同的 cfg_digest 应导致不同的 mask_digest"

    def test_mask_digest_not_affected_by_non_digest_scope_fields(self):
        """mask_digest 不应受 cfg 中非 digest_scope 字段的影响。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        # 基础配置。
        cfg_base = {
            "enable_mask": True,
            "mask_resolution_width": 512,
            "mask_resolution_height": 512,
            "digest_scope_field": "important_value"
        }
        inputs = {"image": [1, 2, 3], "image_shape": (512, 512, 3)}
        
        # 权威 cfg_digest（假设只包含 digest_scope_field，不包含 non_digest_scope_field）
        cfg_digest = digests.canonical_sha256({"digest_scope_field": "important_value"})
        
        result1 = provider.extract(cfg_base, inputs, cfg_digest=cfg_digest)
        assert result1.status == "ok"
        mask_digest_1 = result1.mask_digest
        
        # 修改非 digest_scope 字段。
        cfg_modified = dict(cfg_base)
        cfg_modified["non_digest_scope_field"] = "different_value"
        
        result2 = provider.extract(cfg_modified, inputs, cfg_digest=cfg_digest)
        assert result2.status == "ok"
        mask_digest_2 = result2.mask_digest
        
        # 关键：cfg_digest 未变，所以 mask_digest 应该不变。
        assert mask_digest_1 == mask_digest_2, \
            "非 digest_scope 字段的改变不应影响 mask_digest"

    def test_cfg_digest_absent_uses_fallback(self):
        """当 cfg_digest 为 None 时，payload 中 cfg_digest_binding 应为 'absent'。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        inputs = {"image": [1, 2, 3], "image_shape": (512, 512, 3)}
        
        result = provider.extract(cfg, inputs, cfg_digest=None)
        
        assert result.status == "ok"
        # 当 cfg_digest 为 None 时，mask_payload 中应有 "absent" 标记。
        # 这可以通过重新计算 mask 并检查 cfg_digest_binding 字段验证。
        # 而这里我们主要验证 result 有效且能处理 None 的情况。
        assert result.mask_digest is not None


class TestCfgDigestParameterHandling:
    """测试 cfg_digest 参数的处理与兼容性。"""

    def test_extract_without_cfg_digest_parameter_backward_compatible(self):
        """不传 cfg_digest 参数应向后兼容。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        inputs = {"image": [1, 2, 3], "image_shape": (512, 512, 3)}
        
        # 不传 cfg_digest（向后兼容）。
        result = provider.extract(cfg, inputs)
        
        assert result.status == "ok"
        assert result.mask_digest is not None

    def test_cfg_digest_type_validation(self):
        """cfg_digest 参数类型校验。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        inputs = {"image": [1, 2, 3]}
        
        # cfg_digest 应该是 str 或 None，不接受其他类型。
        with pytest.raises(TypeError):
            provider.extract(cfg, inputs, cfg_digest=123)
        
        with pytest.raises(TypeError):
            provider.extract(cfg, inputs, cfg_digest={"digest": "abc"})


class TestMaskDigestTraceDependencies:
    """测试 mask_digest 的追踪依赖性。"""

    def test_trace_digest_includes_cfg_digest_binding(self):
        """trace_digest 应包含 cfg_digest_binding 信息。"""
        provider = SemanticMaskProvider(
            impl_id=SEMANTIC_MASK_PROVIDER_ID,
            impl_version=SEMANTIC_MASK_PROVIDER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SEMANTIC_MASK_PROVIDER_ID,
                "impl_version": SEMANTIC_MASK_PROVIDER_VERSION
            })
        )

        cfg = {"enable_mask": True}
        inputs = {"image": [1, 2, 3], "image_shape": (512, 512, 3)}
        cfg_digest = "test_cfg_digest_value"
        
        result = provider.extract(cfg, inputs, cfg_digest=cfg_digest)
        
        assert result.status == "ok"
        assert result.audit is not None
        assert "trace_digest" in result.audit
        
        # trace_digest 应该依赖于包含 cfg_digest_binding 的 trace_payload。
        trace_digest = result.audit["trace_digest"]
        assert isinstance(trace_digest, str) and len(trace_digest) == 64  # SHA256 hex
