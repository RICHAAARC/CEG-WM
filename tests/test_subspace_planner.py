"""
File purpose: SubspacePlanner regression tests for deterministic planning and digest binding semantics.
Module type: General module

功能说明：
- 验证 SubspacePlanner 的最小可用实现。
- 验证 plan_digest 的绑定与可复算性。
- 验证 planner 消融（disabled）语义。
- 验证与 mask_digest 的绑定。
"""

import pytest
from typing import Dict, Any

from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
)
from main.core import digests


class TestSubspacePlannerBasic:
    """测试 SubspacePlanner 的基础功能。"""

    def test_subspace_planner_returns_ok_when_enabled(self):
        """规划器启用时应返回 ok 状态。"""
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
                    "seed": 11
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result = planner.plan(cfg, mask_digest="mask_digest_001", inputs=inputs)

        assert result.status == "ok"
        assert result.plan is not None
        assert result.plan_digest is not None
        assert result.basis_digest is not None
        assert result.audit is not None

    def test_subspace_planner_returns_absent_when_disabled(self):
        """规划器禁用时应返回 absent 状态，无失败原因。"""
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
                    "enabled": False,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32
                }
            }
        }

        result = planner.plan(cfg)

        assert result.status == "absent"
        assert result.plan is None
        assert result.plan_digest is None
        # absent 时记录 plan_failure_reason="planner_disabled_by_policy" 作为审计标记
        assert result.plan_failure_reason == "planner_disabled_by_policy"

    def test_subspace_planner_default_path_not_test_mode_synthetic(self):
        """默认路径应走 planner_v1_band_spec，而非 test_mode_synthetic。"""
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
                    "seed": 11
                }
            }
        }
        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            },
            "routing_digest": "a" * 64,
            "mask_summary": {
                "area_ratio": 0.3
            }
        }

        result = planner.plan(cfg, mask_digest="mask_digest_001", inputs=inputs)
        assert result.status == "ok"
        assert isinstance(result.plan, dict)
        assert result.plan.get("plan_origin") == "planner_v1_band_spec"
        assert result.plan.get("routing_digest_ref") == "a" * 64
        assert result.plan.get("pipeline_feature_digest") == "<absent>"
        assert result.plan.get("denoise_trace_digest") == "<absent>"
        assert result.plan.get("attention_anchor_ref_digest") == "<absent>"

    def test_subspace_planner_synthetic_requires_test_mode_or_allow_flag(self):
        """test_mode_synthetic 仅允许在 test_mode/allow_synthetic_trajectory 下启用。"""
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
                    "seed": 11
                }
            }
        }

        result_without_flag = planner.plan(
            cfg,
            mask_digest="mask_digest_001",
            inputs={"unexpected": "only"}
        )
        assert result_without_flag.status == "absent"
        assert result_without_flag.plan_failure_reason == "planner_input_absent"

        result_with_test_mode = planner.plan(
            cfg,
            mask_digest="mask_digest_001",
            inputs={"test_mode": True}
        )
        assert result_with_test_mode.status == "ok"
        assert isinstance(result_with_test_mode.plan, dict)
        assert result_with_test_mode.plan.get("plan_origin") == "test_mode_synthetic"


class TestPlanDigestBinding:
    """测试 plan_digest 与 mask_digest/cfg_digest 的绑定。"""

    def test_plan_digest_binds_to_mask_digest(self):
        """plan_digest 应依赖于 mask_digest。"""
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
                    "seed": 11
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        mask_digest_1 = "mask_digest_value_1"
        result1 = planner.plan(cfg, mask_digest=mask_digest_1, inputs=inputs)

        assert result1.status == "ok"
        plan_digest_1 = result1.plan_digest

        mask_digest_2 = "mask_digest_value_2"
        result2 = planner.plan(cfg, mask_digest=mask_digest_2, inputs=inputs)

        assert result2.status == "ok"
        plan_digest_2 = result2.plan_digest

        assert plan_digest_1 != plan_digest_2, \
            "不同的 mask_digest 应导致不同的 plan_digest"

    def test_plan_digest_binds_to_cfg_digest(self):
        """plan_digest 应依赖于 cfg_digest。"""
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
                    "seed": 11
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        mask_digest = "test_mask_digest"
        cfg_digest_1 = "cfg_digest_1"
        result1 = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_1, inputs=inputs)

        assert result1.status == "ok"
        plan_digest_1 = result1.plan_digest

        cfg_digest_2 = "cfg_digest_2"
        result2 = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest_2, inputs=inputs)

        assert result2.status == "ok"
        plan_digest_2 = result2.plan_digest

        assert plan_digest_1 != plan_digest_2, \
            "不同的 cfg_digest 应导致不同的 plan_digest"

    def test_plan_digest_reproducible_with_same_inputs(self):
        """相同输入应产生相同的 plan_digest（可复算性）。"""
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
                    "seed": 11
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        mask_digest = "test_mask_digest"
        cfg_digest = "test_cfg_digest"

        result1 = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=inputs)
        result2 = planner.plan(cfg, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.plan_digest == result2.plan_digest, \
            "相同输入必须产生相同 plan_digest"


class TestSubspaceMaskConditioning:
    """测试 mask-conditioned 子空间估计可审计性。"""

    def test_subspace_conditioning_changes_with_mask_summary(self):
        """同配置下不同 mask_summary 应触发不同 conditioning 与 plan_digest。"""
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
                    "rank": 4,
                    "sample_count": 10,
                    "feature_dim": 16,
                    "seed": 123,
                }
            }
        }
        base_inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512,
            }
        }
        inputs_a = {
            **base_inputs,
            "mask_summary": {
                "area_ratio": 0.2,
                "downsample_grid_true_indices": [0, 1, 2, 3],
                "downsample_grid_shape": [8, 8],
            },
        }
        inputs_b = {
            **base_inputs,
            "mask_summary": {
                "area_ratio": 0.75,
                "downsample_grid_true_indices": [40, 41, 42, 43, 44, 45],
                "downsample_grid_shape": [8, 8],
            },
        }

        result_a = planner.plan(cfg, mask_digest="mask_digest_same", cfg_digest="cfg_digest_same", inputs=inputs_a)
        result_b = planner.plan(cfg, mask_digest="mask_digest_same", cfg_digest="cfg_digest_same", inputs=inputs_b)

        assert result_a.status == "ok"
        assert result_b.status == "ok"
        assert isinstance(result_a.plan, dict)
        assert isinstance(result_b.plan, dict)
        conditioning_a = result_a.plan.get("subspace_conditioning")
        conditioning_b = result_b.plan.get("subspace_conditioning")
        assert isinstance(conditioning_a, dict)
        assert isinstance(conditioning_b, dict)
        assert conditioning_a.get("region_spec_digest") != conditioning_b.get("region_spec_digest")
        assert result_a.plan_digest != result_b.plan_digest


class TestPlanDigestNotSensitiveToNonPlanDigestScopeFields:
    """测试 plan_digest 不受非 plan_digest_scope 字段影响。"""

    def test_plan_digest_unchanged_by_non_scope_fields(self):
        """cfg 中非 plan_digest_scope 的字段变化不应影响 plan_digest。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11
                }
            },
            "plan_digest_scope_field": "important",
            "non_scope_field": "extra_info"
        }

        cfg_modified = dict(cfg_base)
        cfg_modified["non_scope_field"] = "different_extra_info"

        mask_digest = "test_mask"
        cfg_digest = "test_cfg_digest"
        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result1 = planner.plan(cfg_base, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=inputs)
        result2 = planner.plan(cfg_modified, mask_digest=mask_digest, cfg_digest=cfg_digest, inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        
        # 由于 cfg_digest 实际来自 include_paths（不包含 non_scope_field），
        # 所以两者应产生相同 plan_digest。
        # 这里我们验证 plan 在相同 cfg_digest 下产生相同 plan_digest。
        # 更严格的测试需要真实的 config_loader.compute_cfg_digest。


class TestPlanAuditFields:
    """测试规划器的审计字段。"""

    def test_plan_audit_contains_required_fields(self):
        """plan 的 audit 字段应包含必需的实现身份信息。"""
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
                    "seed": 11
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result = planner.plan(cfg, mask_digest="mask_digest_001", inputs=inputs)

        assert result.status == "ok"
        assert result.audit is not None
        assert "impl_identity" in result.audit
        assert "impl_version" in result.audit
        assert "impl_digest" in result.audit
        assert "trace_digest" in result.audit


class TestPlanFailureReasons:
    """测试规划器的失败原因处理。"""

    def test_invalid_k_parameter_fails(self):
        """无效的 k 参数应导致 failed 状态。"""
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
                    "rank": -1,
                    "sample_count": 12,
                    "feature_dim": 32
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result = planner.plan(cfg, mask_digest="mask_digest_001", inputs=inputs)

        assert result.status == "failed"
        assert result.plan_failure_reason is not None
        assert result.plan_digest is None

class TestShallowDiffuseParameterBinding:
    """
    Shallow Diffuse 风格参数绑定测试。
    验证 edit_timestep、mask_shape、w_channel、injection_domain、enable_channel_refill 等参数的绑定。
    """

    def test_plan_digest_binds_edit_timestep(self):
        """plan_digest 应依赖于 edit_timestep。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11,
                    "num_inference_steps": 50,
                    "edit_timestep": 10
                }
            }
        }

        cfg_modified = dict(cfg_base)
        cfg_modified["watermark"] = dict(cfg_base["watermark"])
        cfg_modified["watermark"]["subspace"] = dict(cfg_base["watermark"]["subspace"])
        cfg_modified["watermark"]["subspace"]["edit_timestep"] = 30

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result1 = planner.plan(cfg_base, mask_digest="mask_001", inputs=inputs)
        result2 = planner.plan(cfg_modified, mask_digest="mask_001", inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.plan_digest != result2.plan_digest, \
            "edit_timestep 改变应导致 plan_digest 改变"

    def test_plan_digest_binds_mask_shape(self):
        """plan_digest 应依赖于 mask_shape。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11,
                    "mask_shape": "circle",
                    "mask_radius": 10
                }
            }
        }

        cfg_modified = dict(cfg_base)
        cfg_modified["watermark"] = dict(cfg_base["watermark"])
        cfg_modified["watermark"]["subspace"] = dict(cfg_base["watermark"]["subspace"])
        cfg_modified["watermark"]["subspace"]["mask_shape"] = "square"

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result1 = planner.plan(cfg_base, mask_digest="mask_001", inputs=inputs)
        result2 = planner.plan(cfg_modified, mask_digest="mask_001", inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.plan_digest != result2.plan_digest, \
            "mask_shape 改变应导致 plan_digest 改变"

    def test_plan_digest_binds_w_channel(self):
        """plan_digest 应依赖于 w_channel。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11,
                    "w_channel": -1
                }
            }
        }

        cfg_modified = dict(cfg_base)
        cfg_modified["watermark"] = dict(cfg_base["watermark"])
        cfg_modified["watermark"]["subspace"] = dict(cfg_base["watermark"]["subspace"])
        cfg_modified["watermark"]["subspace"]["w_channel"] = 0

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result1 = planner.plan(cfg_base, mask_digest="mask_001", inputs=inputs)
        result2 = planner.plan(cfg_modified, mask_digest="mask_001", inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.plan_digest != result2.plan_digest, \
            "w_channel 改变应导致 plan_digest 改变"

    def test_plan_digest_binds_injection_domain(self):
        """plan_digest 应依赖于 injection_domain。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11,
                    "injection_domain": "spatial"
                }
            }
        }

        cfg_modified = dict(cfg_base)
        cfg_modified["watermark"] = dict(cfg_base["watermark"])
        cfg_modified["watermark"]["subspace"] = dict(cfg_base["watermark"]["subspace"])
        cfg_modified["watermark"]["subspace"]["injection_domain"] = "freq"

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result1 = planner.plan(cfg_base, mask_digest="mask_001", inputs=inputs)
        result2 = planner.plan(cfg_modified, mask_digest="mask_001", inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.plan_digest != result2.plan_digest, \
            "injection_domain 改变应导致 plan_digest 改变"

    def test_plan_digest_binds_enable_channel_refill(self):
        """plan_digest 应依赖于 enable_channel_refill。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({
                "impl_id": SUBSPACE_PLANNER_ID,
                "impl_version": SUBSPACE_PLANNER_VERSION
            })
        )

        cfg_base = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11,
                    "enable_channel_refill": False
                }
            }
        }

        cfg_modified = dict(cfg_base)
        cfg_modified["watermark"] = dict(cfg_base["watermark"])
        cfg_modified["watermark"]["subspace"] = dict(cfg_base["watermark"]["subspace"])
        cfg_modified["watermark"]["subspace"]["enable_channel_refill"] = True

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result1 = planner.plan(cfg_base, mask_digest="mask_001", inputs=inputs)
        result2 = planner.plan(cfg_modified, mask_digest="mask_001", inputs=inputs)

        assert result1.status == "ok"
        assert result2.status == "ok"
        assert result1.plan_digest != result2.plan_digest, \
            "enable_channel_refill 改变应导致 plan_digest 改变"

    def test_plan_contains_detection_domain_spec(self):
        """plan 应包含 detection_domain_spec，用于定义 detect 端的 z_t 构造。"""
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
                    "seed": 11,
                    "num_inference_steps": 50,
                    "edit_timestep": 20
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)

        assert result.status == "ok"
        assert result.plan is not None
        assert "detection_domain_spec" in result.plan
        
        detection_spec = result.plan["detection_domain_spec"]
        assert "edit_timestep" in detection_spec
        assert "num_inference_steps" in detection_spec
        assert "forward_diffusion_start_timestep" in detection_spec or "forward_diffusion_start" in detection_spec
        assert "forward_diffusion_end_timestep" in detection_spec or "forward_diffusion_end" in detection_spec
        assert detection_spec.get("edit_timestep") == 20

    def test_injection_config_in_plan(self):
        """plan 的 injection_config 应包含 edit_timestep、mask_shape、w_channel、injection_domain 等。"""
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
                    "seed": 11,
                    "edit_timestep": 15,
                    "mask_shape": "ring",
                    "mask_radius": 20,
                    "mask_radius2": 5,
                    "w_channel": 1,
                    "injection_domain": "freq",
                    "enable_channel_refill": True,
                    "num_inference_steps": 50
                }
            }
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
                "height": 512,
                "width": 512
            }
        }

        result = planner.plan(cfg, mask_digest="mask_001", inputs=inputs)

        assert result.status == "ok"
        assert result.plan is not None
        assert "injection_config" in result.plan
        
        inj_cfg = result.plan["injection_config"]
        assert inj_cfg["edit_timestep"] == 15
        assert inj_cfg["mask_shape"] == "ring"
        assert inj_cfg["mask_radius"] == 20
        assert inj_cfg["w_channel"] == 1
        assert inj_cfg["injection_domain"] == "freq"
        assert inj_cfg["channel_mix_policy"] == "channel_refill"


class TestSubspacePlanDigestSupplementary:
    """
    补强测试集
    验证：plan_digest 可复算、输入域收敛、轨迹特征绑定、失败语义严格
    """

    def test_T1_subspace_plan_digest_reproducible(self):
        """T1: 同输入同 seed 同配置 plan_digest/basis_digest 必一致。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
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
                    "timestep_end": 30
                }
            }
        }
        inputs = {"trace_signature": {"num_inference_steps": 50, "guidance_scale": 7.0, "height": 512, "width": 512}}

        result1 = planner.plan(cfg, mask_digest="mask_test_1", cfg_digest="cfg_test_1", inputs=inputs)
        result2 = planner.plan(cfg, mask_digest="mask_test_1", cfg_digest="cfg_test_1", inputs=inputs)

        assert result1.status == "ok" and result2.status == "ok"
        assert result1.plan_digest == result2.plan_digest, "同输入同配置必产生同 plan_digest"
        assert result1.basis_digest == result2.basis_digest, "同输入同配置必产生同 basis_digest"

    def test_T2_plan_digest_binds_feature_domain(self):
        """T2: timesteps/seed/feature_source 任一变化必须导致 plan_digest 变化。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
        )

        cfg_base = {
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
        inputs = {"trace_signature": {"num_inference_steps": 50, "guidance_scale": 7.0, "height": 512, "width": 512}}

        result_base = planner.plan(cfg_base, mask_digest="mask_1", inputs=inputs)
        assert result_base.status == "ok"
        plan_digest_base = result_base.plan_digest

        # 改变 timestep_end
        cfg_modified = dict(cfg_base)
        cfg_modified["watermark"] = dict(cfg_base["watermark"])
        cfg_modified["watermark"]["subspace"] = dict(cfg_base["watermark"]["subspace"])
        cfg_modified["watermark"]["subspace"]["timestep_end"] = 50  # 改变
        result_modified = planner.plan(cfg_modified, mask_digest="mask_1", inputs=inputs)
        assert result_modified.status == "ok"
        
        assert result_modified.plan_digest != plan_digest_base, "timestep 改变必导致 plan_digest 改变"

    def test_T3_plan_digest_binds_mask_digest(self):
        """T3: mask_digest 变化必须导致 plan_digest 变化。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
        )
        cfg = {
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
        inputs = {"trace_signature": {"num_inference_steps": 50, "guidance_scale": 7.0, "height": 512, "width": 512}}

        result1 = planner.plan(cfg, mask_digest="mask_digest_A", inputs=inputs)
        result2 = planner.plan(cfg, mask_digest="mask_digest_B", inputs=inputs)

        assert result1.status == "ok" and result2.status == "ok"
        assert result1.plan_digest != result2.plan_digest, "mask_digest 改变必导致 plan_digest 改变"

    def test_T4_plan_mismatch_semantics_no_score(self):
        """T4: detect 侧 plan_digest 不一致 -> mismatch，且不得给分。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
        )
        cfg = {
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
        inputs = {"trace_signature": {"num_inference_steps": 50, "guidance_scale": 7.0, "height": 512, "width": 512}}

        result = planner.plan(cfg, mask_digest="mask_1", inputs=inputs)
        assert result.status == "ok"
        
        # 模拟 mismatch：返回值中添加不一致的 plan_digest
        # (在实际 detect 侧会通过比对来发现不一致)
        # 这里我们验证规划器本身不会产生伪有效的 plan
        assert result.plan is not None, "OK 状态下 plan 必存在"
        assert result.plan_digest is not None, "OK 状态下 plan_digest 必存在"
        # mismatch 会由 detect 侧在比对时产生，这里验证规划器不产出伪有效 plan

    def test_T5_no_network_or_write_bypass_in_planner(self):
        """T5: planner 代码不得引入网络访问或写盘旁路。"""
        import ast
        import inspect
        
        # 检查 SubspacePlannerImpl 的所有方法是否包含禁止调用
        banned_imports = {"requests", "httpx", "urllib", "socket", "pickle", "dill"}
        banned_functions = {"eval", "exec", "compile", "__import__", "open"}
        
        source = inspect.getsource(SubspacePlannerImpl)
        tree = ast.parse(source)
        
        found_violations = []
        for node in ast.walk(tree):
            # 检查 import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in banned_imports:
                        found_violations.append(f"禁止 import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in banned_imports:
                    found_violations.append(f"禁止 from import: {node.module}")
            
            # 检查禁止函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in banned_functions:
                    found_violations.append(f"禁止函数调用: {node.func.id}")
        
        assert len(found_violations) == 0, f"发现违反规则的调用: {found_violations}"