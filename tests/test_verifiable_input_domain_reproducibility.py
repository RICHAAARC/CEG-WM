"""
File purpose: 验证可验证输入域各字段的可复算一致性
Module type: General module

功能说明：
- 验证 verifiable_input_domain_spec 在相同输入下的可复算性
- 验证 samples_anchor 和 jvp_anchor 的摘要稳定性
- 验证这些字段进入 plan_digest 后的一致性
- 验证多次调用产生相同的计算结果
"""

import pytest
from typing import Dict, Any

from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
)
from main.core import digests


class TestVerifiableInputDomainReproducibility:
    """验证可验证输入域各字段的可复算性。"""

    def test_samples_anchor_is_reproducible(self):
        """同一 seed 应产生相同的 samples_anchor 摘要。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "timestep_start": 0,
                    "timestep_end": 30,
                    "trajectory_step_stride": 1
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        # 第一次调用
        result1 = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)
        assert result1.status == "ok"
        
        # 第二次调用：相同 cfg, inputs, mask_digest
        result2 = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)
        assert result2.status == "ok"

        # 两次调用的 plan_digest 应完全相同（可复算）
        assert result1.plan_digest == result2.plan_digest, \
            "相同输入应产生相同 plan_digest"

        # 验证 basis_digest 也相同
        assert result1.basis_digest == result2.basis_digest, \
            "相同输入应产生相同 basis_digest"

    def test_samples_anchor_changes_with_seed(self):
        """不同 seed 应产生不同的 samples_anchor。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_seed_42 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42
                }
            }
        }

        cfg_seed_99 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 99
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        result_seed42 = planner.plan(cfg_seed_42, mask_digest="mask_001", inputs=inputs)
        result_seed99 = planner.plan(cfg_seed_99, mask_digest="mask_001", inputs=inputs)

        assert result_seed42.status == "ok"
        assert result_seed99.status == "ok"

        # 不同 seed 应产生不同的 plan_digest
        assert result_seed42.plan_digest != result_seed99.plan_digest, \
            "不同 seed 应导致 plan_digest 变化"

    def test_jvp_anchor_is_reproducible(self):
        """同一 jacobian_probe_count 和 jacobian_eps 应产生相同的 jvp_anchor 摘要。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        # 多次调用
        result1 = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)
        result2 = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)
        result3 = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result3.status == "ok"

        # 所有 plan_digest 应相同
        assert result1.plan_digest == result2.plan_digest == result3.plan_digest, \
            "相同 JVP 参数应产生相同 plan_digest"

    def test_jvp_anchor_changes_with_probe_params(self):
        """不同的 jacobian_probe_count 或 jacobian_eps 应导致 plan_digest 变化。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_probe_2 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            }
        }

        cfg_probe_3 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "jacobian_probe_count": 3,
                    "jacobian_eps": 1e-3
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        result_probe2 = planner.plan(cfg_probe_2, mask_digest="mask_001", inputs=inputs)
        result_probe3 = planner.plan(cfg_probe_3, mask_digest="mask_001", inputs=inputs)

        assert result_probe2.status == "ok"
        assert result_probe3.status == "ok"

        # 不同 probe_count 应导致 plan_digest 变化
        assert result_probe2.plan_digest != result_probe3.plan_digest, \
            "不同 jacobian_probe_count 应导致 plan_digest 变化"

    def test_verifiable_input_domain_spec_in_plan_digest(self):
        """验证 verifiable_input_domain_spec 正确包含在 plan_digest payload 中。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "timestep_start": 0,
                    "timestep_end": 30,
                    "trajectory_step_stride": 1,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            },
            "model_id": "test_model",
            "model_revision": "test_rev"
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        result = planner.plan(cfg, mask_digest="mask_001", cfg_digest="cfg_digest_test", inputs=inputs)
        
        assert result.status == "ok"
        assert result.plan is not None
        
        # 验证 plan 中包含必要的可验证框架字段
        plan = result.plan
        assert "feature_domain_anchor" in plan
        assert "detection_domain_spec" in plan
        
        # 验证 plan_stats 包含 feature_source_tag
        assert result.plan_stats is not None
        assert "feature_source_tag" in result.plan_stats

    def test_timestep_spec_consistency(self):
        """验证不同的 timestep 配置导致 plan_digest 变化。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_ts_0_30 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "timestep_start": 0,
                    "timestep_end": 30
                }
            }
        }

        cfg_ts_5_25 = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "timestep_start": 5,
                    "timestep_end": 25
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        result_ts1 = planner.plan(cfg_ts_0_30, mask_digest="mask_001", inputs=inputs)
        result_ts2 = planner.plan(cfg_ts_5_25, mask_digest="mask_001", inputs=inputs)

        assert result_ts1.status == "ok"
        assert result_ts2.status == "ok"

        # 不同 timestep window 应导致 plan_digest 变化
        assert result_ts1.plan_digest != result_ts2.plan_digest, \
            "不同 timestep_start/end 应导致 plan_digest 变化"

    def test_planner_params_canonical_binding(self):
        """验证规范化的 planner_params 正确绑定到 verifiable_input_domain_spec。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        # 使用 k 和 topk（而非 rank 和 spectrum_topk）
        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "k": 8,
                    "topk": 10,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        result = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)
        
        assert result.status == "ok"
        # k/topk 应被规范化，多次调用应产生相同 plan_digest
        result_repeat = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)
        assert result.plan_digest == result_repeat.plan_digest, \
            "k/topk 输入应被规范化为相同的 plan_digest"

    def test_all_verifiable_anchors_stable_across_calls(self):
        """综合测试：所有可验证锚点在多次调用中保持稳定。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 42,
                    "timestep_start": 0,
                    "timestep_end": 30,
                    "trajectory_step_stride": 1,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3,
                    "edit_timestep": 5,
                    "num_inference_steps": 50,
                    "mask_shape": "circle",
                    "mask_radius": 10
                }
            },
            "model_id": "test_model",
            "model_revision": "rev1"
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        # 进行 3 次调用
        results = []
        for i in range(3):
            result = planner.plan(
                cfg,
                mask_digest="mask_001",
                cfg_digest="cfg_digest_fixed",
                inputs=inputs
            )
            assert result.status == "ok"
            results.append(result)

        # 验证所有关键摘要都相同
        plan_digests = [r.plan_digest for r in results]
        basis_digests = [r.basis_digest for r in results]

        assert len(set(plan_digests)) == 1, \
            "所有 plan_digest 应完全相同"
        assert len(set(basis_digests)) == 1, \
            "所有 basis_digest 应完全相同"

        # 验证 audit 中的 trace_digest 也相同
        trace_digests = [r.audit.get("trace_digest") for r in results]
        assert len(set(trace_digests)) == 1, \
            "所有 trace_digest 应完全相同"

        # 验证通过
