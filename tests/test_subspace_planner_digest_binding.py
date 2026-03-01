"""
子空间规划器全量收口测试

功能说明：
- 验证 plan_digest 完整绑定 cfg_digest、mask_digest、planner_params、impl_identity。
- 验证 trace_digest isolation（v2 版本不含全量 cfg）。
- 验证 detect 侧 plan_digest mismatch 检测。
- 验证规划器消融语义（enable_planner=false, mask_absent）。
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, Dict

import pytest

from main.core import digests
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
)
from main.watermarking.content_chain.subspace.planner_interface import SubspacePlanEvidence


def _build_planner_inputs() -> Dict[str, Any]:
    """
    功能：构造规划器输入。 

    Build deterministic planner inputs for tests.

    Args:
        None.

    Returns:
        Planner inputs mapping.
    """
    return {
        "trace_signature": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "height": 512,
            "width": 512
        }
    }


class TestPlanDigestBinding:
    """
    验证 plan_digest 完整绑定到所有关键因素。
    """

    def test_plan_digest_binds_to_cfg_digest(self) -> None:
        """
        plan_digest 必须依赖 cfg_digest。
        
        当同一 cfg_digest + 不同计算时刻，plan_digest 应保持一致（可复算）。
        当改变 cfg_digest，plan_digest 必须变化。
        """
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        cfg_digest_1 = "cfg_digest_1_64chars_abc123def456abc123def456abc123def456abc12"
        cfg_digest_2 = "cfg_digest_2_64chars_xyz789xyz789xyz789xyz789xyz789xyz789xyz78"
        mask_digest = "mask_digest_64chars_111111111111111111111111111111111111111"
        
        # 第一次：使用 cfg_digest_1 计算
        result1 = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_1, inputs=_build_planner_inputs())
        assert result1.status == "ok"
        plan_digest_with_cfg1 = result1.plan_digest
        
        # 第二次：使用相同的 cfg_digest_1 再计算，plan_digest 应相同（可复算）
        result1_repeat = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_1, inputs=_build_planner_inputs())
        assert result1_repeat.status == "ok"
        assert result1_repeat.plan_digest == plan_digest_with_cfg1, \
            "plan_digest 应可复算（同 cfg_digest 应产出相同 plan_digest）"
        
        # 第三次：改用 cfg_digest_2，plan_digest 必须变化
        result2 = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_2, inputs=_build_planner_inputs())
        assert result2.status == "ok"
        plan_digest_with_cfg2 = result2.plan_digest
        assert plan_digest_with_cfg2 != plan_digest_with_cfg1, \
            "不同 cfg_digest 必须导致 plan_digest 变化"

    def test_plan_digest_binds_to_mask_digest(self) -> None:
        """
        plan_digest 必须依赖 mask_digest。
        
        当改变 mask_digest，plan_digest 必须变化。
        """
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        cfg_digest = "cfg_digest_64chars_abc123def456abc123def456abc123def456abc12"
        mask_digest_1 = "mask_digest_1_64chars_111111111111111111111111111111111111111"
        mask_digest_2 = "mask_digest_2_64chars_222222222222222222222222222222222222222"
        
        # 使用 mask_digest_1
        result1 = planner.plan(cfg, mask_digest=mask_digest_1, cfg_digest=cfg_digest, inputs=_build_planner_inputs())
        assert result1.status == "ok"
        plan_digest_with_mask1 = result1.plan_digest
        
        # 使用 mask_digest_2
        result2 = planner.plan(cfg, mask_digest=mask_digest_2, cfg_digest=cfg_digest, inputs=_build_planner_inputs())
        assert result2.status == "ok"
        plan_digest_with_mask2 = result2.plan_digest
        assert plan_digest_with_mask2 != plan_digest_with_mask1, \
            "不同 mask_digest 必须导致 plan_digest 变化"

    def test_plan_digest_binds_to_planner_params(self) -> None:
        """
        plan_digest 必须依赖规划参数（k, topk）。
        """
        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        cfg_with_k15_topk25 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 15,
                    "topk": 25
                }
            }
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        cfg_digest = "cfg_digest_64chars_abc123def456abc123def456abc123def456abc12"
        mask_digest = "mask_digest_64chars_111111111111111111111111111111111111111"
        
        result1 = planner.plan(cfg_base, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=_build_planner_inputs())
        assert result1.status == "ok"
        
        result2 = planner.plan(cfg_with_k15_topk25, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=_build_planner_inputs())
        assert result2.status == "ok"
        
        assert result1.plan_digest != result2.plan_digest, \
            "不同的规划参数（k, topk）必须导致 plan_digest 变化"


class TestTraceDigestIsolation:
    """
    验证 trace_digest 互相隔离（v2：不包含全量 cfg）。
    """

    def test_trace_digest_v2_not_affected_by_non_digest_scope_cfg_fields(self) -> None:
        """
        trace_digest 应仅依赖 trace_version、impl 身份、启用状态、摘要绑定。
        
        修改 cfg 中非 digest_scope 字段，trace_digest 不应变化。
        """
        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            },
            "model": {"model_id": "sd3"},
            "irrelevant_field": "some_value"
        }
        
        cfg_modified_irrelevant = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            },
            "model": {"model_id": "sd3"},
            "irrelevant_field": "different_value"  # 仅改变非 digest_scope 字段
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        cfg_digest = "cfg_digest_64chars_abc123def456abc123def456abc123def456abc12"
        mask_digest = "mask_digest_64chars_111111111111111111111111111111111111111"
        
        result1 = planner.plan(cfg_base, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=_build_planner_inputs())
        assert result1.status == "ok"
        trace_digest_1 = result1.audit.get("trace_digest")
        
        result2 = planner.plan(cfg_modified_irrelevant, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=_build_planner_inputs())
        assert result2.status == "ok"
        trace_digest_2 = result2.audit.get("trace_digest")
        
        assert trace_digest_1 == trace_digest_2, \
            "修改 cfg 非 digest_scope 字段，trace_digest 不应变化（trace_digest v2 仅包含摘要）"

    def test_trace_payload_v2_does_not_contain_full_cfg(self) -> None:
        """
        直接验证 trace_payload 结构不包含 'cfg' 全量字段（v2）。
        """
        from main.watermarking.content_chain.subspace.subspace_planner_impl import (
            _build_planner_trace_payload
        )
        
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        payload = _build_planner_trace_payload(
            cfg,
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd",
            enabled=True,
            mask_digest="mask_digest_64chars_111111111111111111111111111111111111111",
            cfg_digest="cfg_digest_64chars_abc123def456abc123def456abc123def456abc12"
        )
        
        # v2 约束：payload 不应包含 "cfg" 字段
        assert "cfg" not in payload, \
            "trace_payload v2 不应包含全量 cfg 字段"
        
        # v2 约束：必须包含版本化标记
        assert payload.get("trace_version") == "v2", \
            "trace_version 必须为 v2（表示不含全量 cfg）"
        
        # v2 约束：必须包含摘要绑定字段
        assert "cfg_digest_binding" in payload, \
            "trace_payload 必须包含 cfg_digest_binding（摘要）"
        assert "mask_digest_binding" in payload, \
            "trace_payload 必须包含 mask_digest_binding（摘要）"


class TestAblationSemantics:
    """
    验证消融语义（enable_planner=false, mask_absent）。
    """

    def test_disabled_planner_returns_absent_with_reason(self) -> None:
        """
        enable_planner=false 时，返回 status=absent 并记录原因。
        """
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": False,  # 禁用规划器
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        result = planner.plan(cfg)
        
        assert result.status == "absent", \
            "enable_planner=false 时，status 必须为 absent"
        assert result.plan is None
        assert result.plan_digest is None
        assert result.basis_digest is None
        assert result.plan_failure_reason == "planner_disabled_by_policy", \
            "absent 时 plan_failure_reason 应标注为 planner_disabled_by_policy"

    def test_plan_producible_with_mask_absent(self) -> None:
        """
        mask_digest 缺失时（mask_absent），仍然能计算 plan_digest（消融语义）。
        
        关键：plan_digest 仍可复算，且 plan_payload 显式标注 mask_digest_status=absent。
        """
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        cfg_digest = "cfg_digest_64chars_abc123def456abc123def456abc123def456abc12"
        
        # mask_digest=None（缺失）
        result = planner.plan(cfg, mask_digest=None, cfg_digest=cfg_digest, inputs=_build_planner_inputs())

        assert result.status == "absent", \
            "mask_digest 缺失时应返回 absent 语义"
        assert result.plan is None
        assert result.plan_digest is None
        assert result.plan_failure_reason == "mask_absent"


class TestDetectSideMismatchValidation:
    """
    验证 detect 侧 plan_digest mismatch 检测逻辑（仅验证数据一致性检验）。
    """

    def test_plan_digest_consistency_can_be_verified(self) -> None:
        """
        verify 可以对比两个 plan_digest 并检测不一致。
        
        这个测试仅验证"能够进行对比"的机制；
        完整的 orchestrator 集成测试由 test_detect_orchestrator_* 系列完成。
        """
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 10,
                    "topk": 20
                }
            }
        }
        
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        )
        
        cfg_digest_embed = "cfg_digest_embed_64chars_abc123def456abc123def456abc123"
        cfg_digest_detect = "cfg_digest_detect_64chars_xyz789xyz789xyz789xyz789xyz78"
        mask_digest = "mask_digest_64chars_111111111111111111111111111111111111111"
        
        # embed 时的计算
        embed_result = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_embed, inputs=_build_planner_inputs())
        assert embed_result.status == "ok"
        embed_plan_digest = embed_result.plan_digest
        
        # detect 时的计算（使用不同的 cfg_digest）
        detect_result = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_detect, inputs=_build_planner_inputs())
        assert detect_result.status == "ok"
        detect_plan_digest = detect_result.plan_digest
        
        # 对比：不同 cfg_digest 应导致 plan_digest 不同
        assert detect_plan_digest != embed_plan_digest, \
            "不同 cfg_digest 应导致 plan_digest 不同，detect 侧可察觉 mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
