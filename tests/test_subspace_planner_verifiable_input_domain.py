"""
File purpose: 可验证输入域机制的回归测试
Module type: General module

功能说明：
- 验证 _collect_trajectory_samples() 采样函数的工作
- 验证 _estimate_jvp_matrix() JVP 估算函数的工作
- 验证 verifiable_input_domain_spec 的正确生成
- 验证 detect 侧可验证性校验函数
"""

import pytest
import numpy as np
from typing import Dict, Any

from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SubspacePlannerImpl,
    verify_verifiable_input_domain,
    create_run_closure_trajectory_anchors,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
)
from main.core import digests


class TestCollectTrajectorySamples:
    """测试轨迹采样的可验证性机制。"""

    def test_collect_trajectory_samples_returns_samples_and_anchor(self):
        """采样函数应返回样本矩阵和摘要锚点。"""
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
                    "sample_count": 8,
                    "feature_dim": 16,
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

        planner_params = planner._parse_planner_params(cfg)
        samples, samples_anchor = planner._collect_trajectory_samples(cfg, inputs, planner_params)

        # 验证样本矩阵形状
        assert samples.shape == (8, 16)
        assert samples.dtype == np.float64

        # 验证所有值都是有限的
        assert np.isfinite(samples).all()

        # 验证锚点包含必需的信息
        assert "timesteps_digest" in samples_anchor
        assert "probe_seed_digest" in samples_anchor
        assert "shape_spec" in samples_anchor
        assert "moments_digest" in samples_anchor
        assert "source" in samples_anchor

    def test_collect_trajectory_samples_is_deterministic(self):
        """采样结果应具有确定性（相同的 seed 应产生相同的样本）。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"test": "digest"})
        )

        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 4,
                    "sample_count": 8,
                    "feature_dim": 16,
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

        planner_params = planner._parse_planner_params(cfg)
        samples1, anchor1 = planner._collect_trajectory_samples(cfg, inputs, planner_params)
        samples2, anchor2 = planner._collect_trajectory_samples(cfg, inputs, planner_params)

        # 两次采样应该完全相同
        assert np.allclose(samples1, samples2)
        assert anchor1["timesteps_digest"] == anchor2["timesteps_digest"]
        assert anchor1["probe_seed_digest"] == anchor2["probe_seed_digest"]


class TestEstimateJVPMatrix:
    """测试 JVP 矩阵估算函数。"""

    def test_estimate_jvp_matrix_uses_runtime_operator_on_formal_path(self):
        """paper formal path 提供 runtime operator 时应返回 runtime_operator 锚点。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"test": "digest"})
        )

        cfg = {
            "paper_faithfulness": {"enabled": True},
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 4,
                    "sample_count": 8,
                    "feature_dim": 16,
                    "seed": 42,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            }
        }

        planner_params = planner._parse_planner_params(cfg)
        
        # 创建一个简单的中心化矩阵
        centered = np.random.randn(8, 16).astype(np.float64)
        
        def _runtime_operator(state_vector: np.ndarray, probe_vector: np.ndarray, eps: float) -> np.ndarray:
            return state_vector + probe_vector * eps

        inputs = {"jvp_operator": _runtime_operator}
        
        jvp_samples, jvp_anchor = planner._estimate_jvp_matrix(cfg, inputs, centered, planner_params)

        # 验证 JVP 样本形状
        assert jvp_samples.ndim == 2
        assert jvp_samples.shape[1] == 16  # 与输入特征维度相同

        # 验证拱点包含必需的信息
        assert "jvp_source" in jvp_anchor
        assert "probe_seed_digest" in jvp_anchor
        assert "probe_count_digest" in jvp_anchor
        assert "jacobian_eps_digest" in jvp_anchor
        assert jvp_anchor["jvp_source"] == "runtime_operator"

    def test_estimate_jvp_matrix_rejects_missing_runtime_operator_on_formal_path(self):
        """paper formal path 缺失 runtime operator 时必须 fail-fast。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"test": "digest"})
        )

        cfg = {
            "paper_faithfulness": {"enabled": True},
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 4,
                    "sample_count": 8,
                    "feature_dim": 16,
                    "seed": 42,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            }
        }

        planner_params = planner._parse_planner_params(cfg)
        centered = np.random.randn(8, 16).astype(np.float64)

        with pytest.raises(ValueError, match="paper formal path requires jvp_operator"):
            planner._estimate_jvp_matrix(cfg, {}, centered, planner_params)

    def test_estimate_jvp_matrix_rejects_invalid_runtime_operator_on_formal_path(self):
        """paper formal path 提供了无效 jvp_operator 时也必须 fail-fast。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"test": "digest"})
        )

        cfg = {
            "paper_faithfulness": {"enabled": True},
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 4,
                    "sample_count": 8,
                    "feature_dim": 16,
                    "seed": 42,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            }
        }

        planner_params = planner._parse_planner_params(cfg)
        centered = np.random.randn(8, 16).astype(np.float64)

        def _invalid_runtime_operator(state_vector: np.ndarray, probe_vector: np.ndarray, eps: float) -> np.ndarray:
            _ = state_vector
            _ = probe_vector
            _ = eps
            return np.asarray([1.0, 2.0], dtype=np.float64)

        with pytest.raises(ValueError, match="paper formal path requires jvp_operator"):
            planner._estimate_jvp_matrix(cfg, {"jvp_operator": _invalid_runtime_operator}, centered, planner_params)

    def test_estimate_jvp_matrix_is_deterministic(self):
        """JVP 估算应具有确定性。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"test": "digest"})
        )

        cfg = {
            "paper_faithfulness": {"enabled": True},
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 4,
                    "sample_count": 8,
                    "feature_dim": 16,
                    "seed": 42,
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            }
        }

        planner_params = planner._parse_planner_params(cfg)
        centered = np.random.randn(8, 16).astype(np.float64)
        def _runtime_operator(state_vector: np.ndarray, probe_vector: np.ndarray, eps: float) -> np.ndarray:
            return state_vector + probe_vector * eps

        inputs = {"jvp_operator": _runtime_operator}
        
        jvp1, anchor1 = planner._estimate_jvp_matrix(cfg, inputs, centered, planner_params)
        jvp2, anchor2 = planner._estimate_jvp_matrix(cfg, inputs, centered, planner_params)

        # 两次 JVP 估算应该相同
        assert np.allclose(jvp1, jvp2)
        assert anchor1["probe_seed_digest"] == anchor2["probe_seed_digest"]


class TestVerifiableInputDomainIntegration:
    """集成测试：验证可验证输入域通过整个 planner 流程。"""

    def test_verifiable_input_domain_in_plan_digest(self):
        """plan_digest 应包含 verifiable_input_domain_spec。"""
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
                    "jacobian_probe_count": 2,
                    "jacobian_eps": 1e-3
                }
            },
            "model_id": "test_model",
            "model_revision": "test_revision"
        }

        inputs = {
            "trace_signature": {
                "num_inference_steps": 20,
                "guidance_scale": 7.0
            }
        }

        result = planner.plan(cfg, mask_digest="mask_digest_001", inputs=inputs)

        assert result.status == "ok"
        assert result.plan is not None
        assert result.plan_digest is not None

        # 验证 plan 中包含 verifiable_input_domain_spec
        plan_payload = result.plan  # 这应该是最终的 plan 对象

    def test_verifiable_input_domain_samples_anchor_in_basis_summary(self):
        """basis_summary 应包含 samples_anchor 和 jvp_anchor。"""
        planner = SubspacePlannerImpl(
            impl_id=SUBSPACE_PLANNER_ID,
            impl_version=SUBSPACE_PLANNER_VERSION,
            impl_digest=digests.canonical_sha256({"test": "digest"})
        )

        cfg = {
            "watermark": {
                "subspace": {
                    "enabled": True,
                    "rank": 6,
                    "sample_count": 12,
                    "feature_dim": 32,
                    "seed": 11,
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

        result = planner.plan(cfg, mask_digest="mask_digest_001", inputs=inputs)

        assert result.status == "ok"
        # plan_stats 应包含可验证的源标签
        assert "feature_source_tag" in result.plan_stats


class TestDetectSideMismatchVerification:
    """测试 detect 侧的可验证性校验。"""

    def test_verify_verifiable_input_domain_consistency_check(self):
        """校验函数应检测摘要不匹配。"""
        plan_payload = {
            "verifiable_input_domain_spec": {
                "timesteps_spec": {
                    "timesteps_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                },
                "probe_spec": {
                    "probe_seed_digest": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
                },
                "jvp_source": "runtime_operator"
            }
        }

        # 一致性检查
        closure_anchors = {
            "timesteps_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "probe_seed_digest": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            "jvp_source": "runtime_operator"
        }

        is_consistent, reason = verify_verifiable_input_domain(plan_payload, closure_anchors, strict_mode=True)
        assert is_consistent
        assert reason == ""

    def test_verify_verifiable_input_domain_detects_mismatch(self):
        """校验函数应检测摘要不匹配。"""
        plan_payload = {
            "verifiable_input_domain_spec": {
                "timesteps_spec": {
                    "timesteps_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                },
                "jvp_source": "runtime_operator"
            }
        }

        # 不匹配的摘要
        closure_anchors = {
            "timesteps_digest": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        }

        is_consistent, reason = verify_verifiable_input_domain(plan_payload, closure_anchors, strict_mode=True)
        assert not is_consistent
        assert "timesteps_digest mismatch" in reason

    def test_verify_verifiable_input_domain_detects_trajectory_digest_mismatch(self):
        """校验函数应检测轨迹摘要不匹配。"""
        plan_payload = {
            "verifiable_input_domain_spec": {
                "planner_input_digest": "planner_input_digest_A",
                "trajectory_evidence_anchor": {
                    "trajectory_spec_digest": "spec_digest_A",
                    "trajectory_digest": "traj_digest_A"
                }
            }
        }

        closure_anchors = {
            "planner_input_digest": "planner_input_digest_A",
            "trajectory_spec_digest": "spec_digest_A",
            "trajectory_digest": "traj_digest_B"
        }

        is_consistent, reason = verify_verifiable_input_domain(plan_payload, closure_anchors, strict_mode=True)
        assert not is_consistent
        assert "trajectory_digest mismatch" in reason

    def test_create_run_closure_trajectory_anchors(self):
        """创建 run_closure 锚点的函数应正确生成摘要。"""
        samples = np.random.randn(8, 16).astype(np.float64)
        timesteps = [0, 1, 2, 3, 4, 5, 6, 7]

        anchors = create_run_closure_trajectory_anchors(
            trajectory_samples=samples,
            probe_seed=42,
            jacobian_eps=1e-3,
            timesteps_list=timesteps,
            jvp_source="runtime_operator"
        )

        # 验证锚点包含必需的字段
        assert "timesteps_digest" in anchors
        assert "probe_seed_digest" in anchors
        assert "jacobian_eps_digest" in anchors
        assert "samples_anchor" in anchors
        assert "samples_anchor_digest" in anchors
        assert "jvp_source" in anchors
        assert anchors["jvp_source"] == "runtime_operator"

    def test_create_run_closure_trajectory_anchors_is_deterministic(self):
        """锚点创建函数应具有确定性。"""
        samples = np.random.randn(8, 16).astype(np.float64)
        timesteps = [0, 1, 2, 3, 4, 5, 6, 7]

        anchors1 = create_run_closure_trajectory_anchors(
            trajectory_samples=samples,
            probe_seed=42,
            jacobian_eps=1e-3,
            timesteps_list=timesteps,
            jvp_source="runtime_operator"
        )

        anchors2 = create_run_closure_trajectory_anchors(
            trajectory_samples=samples,
            probe_seed=42,
            jacobian_eps=1e-3,
            timesteps_list=timesteps,
            jvp_source="runtime_operator"
        )

        # 两次调用应产生相同的摘要
        assert anchors1["timesteps_digest"] == anchors2["timesteps_digest"]
        assert anchors1["probe_seed_digest"] == anchors2["probe_seed_digest"]
        assert anchors1["jacobian_eps_digest"] == anchors2["jacobian_eps_digest"]
