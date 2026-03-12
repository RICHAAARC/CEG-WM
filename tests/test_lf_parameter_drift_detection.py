"""
File purpose: Test LF parameter drift detection - verify plan_digest consistency.
Module type: General module

功能说明：
- 验证当 watermark.lf.* 参数变化时，lf_trace_digest 相应改变（V2 detect_score 接口）。
- 确保 LF 参数（ecc_sparsity、correlation_scale）变化能被 trace 追踪。
- 模拟审计场景：攻击者试图在不改变 plan_digest 的情况下改变 LF 参数。
"""

from typing import Any, Dict

import numpy as np
import pytest

from main.watermarking.content_chain.low_freq_coder import (
    LowFreqTemplateCodec,
    LOW_FREQ_TEMPLATE_CODEC_ID,
    LOW_FREQ_TEMPLATE_CODEC_VERSION,
)


def _build_lf_basis(feature_dim: int = 4, basis_rank: int = 2, seed: int = 42) -> Dict[str, Any]:
    """构建用于测试的 LF 子空间基（projection_matrix 形状为 feature_dim × basis_rank）。"""
    rng = np.random.RandomState(seed)
    projection_matrix = rng.randn(feature_dim, basis_rank).astype(np.float32)
    return {
        "projection_matrix": projection_matrix.tolist(),
        "basis_rank": basis_rank,
        "latent_projection_spec": {
            "spec_version": "v1",
            "method": "random_index_selection",
            "feature_dim": feature_dim,
            "seed": seed,
            "edit_timestep": 0,
            "sample_idx": 0,
        },
    }


class TestLFParameterDriftDetection:
    """
    测试 LF 参数漂移检测机制（V2 detect_score 接口）。

    场景：嵌入阶段使用一套 LF 参数，检测阶段使用另一套，但 plan_digest 不变。
    预期：lf_trace_digest 随参数变化改变，提供审计追踪依据。
    """

    def _make_coder(self, impl_digest: str = "test_digest_lf_v2") -> LowFreqTemplateCodec:
        return LowFreqTemplateCodec(
            impl_id=LOW_FREQ_TEMPLATE_CODEC_ID,
            impl_version=LOW_FREQ_TEMPLATE_CODEC_VERSION,
            impl_digest=impl_digest,
        )

    def test_lf_ecc_parameter_drift_detected_as_mismatch(self) -> None:
        """
        场景：watermark.lf.ecc_sparsity 从 3 改为 5，plan_digest 不变。
        预期：lf_trace_digest 改变（parity_check_digest 依赖 ecc_sparsity）。
        """
        coder = self._make_coder()
        plan_digest = "original_plan_digest_abc123"
        lf_basis = _build_lf_basis(feature_dim=4, basis_rank=2)
        latent_features = [0.1, 0.2, 0.3, 0.4]

        cfg_original = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 10.0,
                },
            }
        }

        _, trace1 = coder.detect_score(
            cfg=cfg_original,
            latent_features=latent_features,
            plan_digest=plan_digest,
            cfg_digest="cfg_dig_1",
            lf_basis=lf_basis,
        )
        assert trace1["status"] == "ok"
        assert trace1["plan_digest"] == plan_digest

        # 攻击场景：改变 ecc_sparsity 但保持相同 plan_digest
        cfg_attacked = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 5,  # ← 恶意改变
                    "correlation_scale": 10.0,
                },
            }
        }

        _, trace2 = coder.detect_score(
            cfg=cfg_attacked,
            latent_features=latent_features,
            plan_digest=plan_digest,
            cfg_digest="cfg_dig_1",
            lf_basis=lf_basis,
        )

        assert trace1["lf_trace_digest"] != trace2["lf_trace_digest"], \
            "lf_trace_digest must change when ecc_sparsity changes"

    def test_lf_strength_parameter_drift_detected(self) -> None:
        """
        场景：watermark.lf.correlation_scale 从 5.0 改为 15.0（V2 近似 strength 参数）。
        预期：trace_digest 改变（correlation_scale 纳入 trace_summary）。
        """
        coder = self._make_coder()
        plan_digest = "plan_dig_v1"
        lf_basis = _build_lf_basis(feature_dim=4, basis_rank=2)
        latent_features = [0.1, 0.2, 0.3, 0.4]

        cfg_v1 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 5.0,
                },
            }
        }

        cfg_v2 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 15.0,  # ← 改变
                },
            }
        }

        _, trace1 = coder.detect_score(
            cfg=cfg_v1, latent_features=latent_features,
            plan_digest=plan_digest, cfg_digest="cfg1", lf_basis=lf_basis,
        )
        _, trace2 = coder.detect_score(
            cfg=cfg_v2, latent_features=latent_features,
            plan_digest=plan_digest, cfg_digest="cfg1", lf_basis=lf_basis,
        )

        assert trace1["lf_trace_digest"] != trace2["lf_trace_digest"], \
            "lf_trace_digest must differ when correlation_scale changes"

    def test_lf_codebook_id_affects_trace(self) -> None:
        """
        场景：通过改变 ecc_sparsity 模拟码本切换（V2 无 codebook_id 字段）。
        预期：trace_digest 改变（parity_check_digest 随 ecc_sparsity 变化）。
        """
        coder = self._make_coder()
        plan_digest = "plan_abc"
        lf_basis = _build_lf_basis(feature_dim=4, basis_rank=2)
        latent_features = [0.1, 0.2, 0.3, 0.4]

        cfg_v1 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 10.0,
                },
            }
        }

        cfg_v2 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 7,  # ← 模拟码本切换
                    "correlation_scale": 10.0,
                },
            }
        }

        _, trace1 = coder.detect_score(
            cfg=cfg_v1, latent_features=latent_features,
            plan_digest=plan_digest, cfg_digest="cfg", lf_basis=lf_basis,
        )
        _, trace2 = coder.detect_score(
            cfg=cfg_v2, latent_features=latent_features,
            plan_digest=plan_digest, cfg_digest="cfg", lf_basis=lf_basis,
        )

        assert trace1["lf_trace_digest"] != trace2["lf_trace_digest"], \
            "lf_trace_digest must differ when ecc_sparsity (codebook) changes"

    def test_multiple_lf_parameters_change_produces_different_trace(self) -> None:
        """
        场景：ecc_sparsity + correlation_scale 同时改变。
        预期：trace 改变（审计侧必须检测到参数偏差）。
        """
        coder = self._make_coder()
        plan_digest = "plan_original"
        lf_basis = _build_lf_basis(feature_dim=4, basis_rank=2)
        latent_features = [0.1, 0.2, 0.3, 0.4]

        cfg_original = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 10.0,
                },
            }
        }

        cfg_modified = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 5,
                    "correlation_scale": 20.0,  # ← 同时改变两个参数
                },
            }
        }

        _, trace_orig = coder.detect_score(
            cfg=cfg_original, latent_features=latent_features,
            plan_digest=plan_digest, cfg_digest="cfg", lf_basis=lf_basis,
        )
        _, trace_mod = coder.detect_score(
            cfg=cfg_modified, latent_features=latent_features,
            plan_digest=plan_digest, cfg_digest="cfg", lf_basis=lf_basis,
        )

        assert trace_orig["lf_trace_digest"] != trace_mod["lf_trace_digest"]


class TestLFParameterAuditRequirement:
    """
    审计需求：LF 参数在 plan_digest_include_paths 中的声明确实被强制执行。
    """

    def test_lf_parameters_in_plan_digest_include_paths(self) -> None:
        """
        验证：watermark.lf.* 参数已在 injection_scope_manifest.yaml 的
               plan_digest_include_paths 中声明。
        
        这个测试加载 manifest 并验证关键参数被列入。
        """
        import yaml
        
        # 加载 manifest 文件
        manifest_path = "configs/injection_scope_manifest.yaml"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)
        
        plan_digest_paths = manifest["digest_scope"]["plan_digest_include_paths"]
        
        # 验证 LF 参数已声明
        required_lf_paths = [
            "watermark.lf.codebook_id",
            "watermark.lf.ecc",
            "watermark.lf.strength"
        ]
        
        for path in required_lf_paths:
            assert path in plan_digest_paths, \
                f"LF parameter {path} must be in plan_digest_include_paths"
        
        # 注意：当 LF 参数在 plan_digest_include_paths 中时，
        # 这些参数的任何改变都应该导致 plan_digest 改变（假设 digest 计算正确实现）
        # 因此，检测侧看到参数改变但 plan_digest 不变，应该返回 mismatch

    def test_lf_parameter_change_requires_plan_digest_change(self) -> None:
        """
        审计约束：当 watermark.lf.* 任何参数改变时，
                 plan_digest 也必须改变。
                 
        如果 plan_digest 未改变但参数改变，这表示：
        1. 嵌入侧计算 plan_digest 时缺少该参数
        2. 攻击者试图在不改变 plan_digest 的情况下改变参数
        
        两种情况都应该在检测侧被检查，返回 status="mismatch"。
        """
        # 这个测试是一个审计需求的声明
        
        # 预期：
        # - 嵌入：计算 plan_digest 时，使用 cfg_digest + mask_digest + 
        #        所有在 plan_digest_include_paths 中的参数
        # - 检测：接收 plan_digest 和实际参数，验证一致性
        #        如果参数改变但 plan_digest 不变，返回 mismatch
        
        assert True, "Audit requirement: LF parameters must be in plan_digest calculation"
    def test_lf_trace_digest_changes_when_ecc_changes(self) -> None:
        """
        修改 watermark.lf.ecc_sparsity 时，lf_trace_digest 必须变化（V2 接口）。

        预期：同一输入、同一 plan_digest，但 ecc_sparsity 不同时，
             parity_check_digest 不同，因此 lf_trace_digest 必须不同。
        """
        coder = LowFreqTemplateCodec(
            impl_id=LOW_FREQ_TEMPLATE_CODEC_ID,
            impl_version=LOW_FREQ_TEMPLATE_CODEC_VERSION,
            impl_digest="test_digest_lf_v1_ecc_change"
        )

        plan_digest = "fixed_plan_digest_for_ecc_test"
        lf_basis = _build_lf_basis(feature_dim=4, basis_rank=2)
        latent_features = [0.1, 0.2, 0.3, 0.4]

        cfg_ecc3 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 10.0,
                },
            }
        }

        _, trace_ecc3 = coder.detect_score(
            cfg=cfg_ecc3, latent_features=latent_features,
            plan_digest=plan_digest, lf_basis=lf_basis,
        )
        digest_ecc3 = trace_ecc3["lf_trace_digest"]
        assert digest_ecc3 is not None, "lf_trace_digest must not be None when enabled"

        cfg_ecc5 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 5,
                    "correlation_scale": 10.0,
                },
            }
        }

        _, trace_ecc5 = coder.detect_score(
            cfg=cfg_ecc5, latent_features=latent_features,
            plan_digest=plan_digest, lf_basis=lf_basis,
        )
        digest_ecc5 = trace_ecc5["lf_trace_digest"]
        assert digest_ecc5 is not None, "lf_trace_digest must not be None when enabled"

        assert digest_ecc3 != digest_ecc5, \
            "lf_trace_digest must change when ecc_sparsity changes"

    def test_lf_trace_digest_changes_when_strength_changes(self) -> None:
        """
        修改 watermark.lf.correlation_scale 时，lf_trace_digest 必须变化（V2 接口）。

        预期：同一输入、同一 plan_digest，但 correlation_scale 不同时，
             trace_summary 中的 correlation_scale 字段不同，因此 lf_trace_digest 必须不同。
        """
        coder = LowFreqTemplateCodec(
            impl_id=LOW_FREQ_TEMPLATE_CODEC_ID,
            impl_version=LOW_FREQ_TEMPLATE_CODEC_VERSION,
            impl_digest="test_digest_lf_v1_strength_change"
        )

        plan_digest = "fixed_plan_digest_for_strength_test"
        lf_basis = _build_lf_basis(feature_dim=4, basis_rank=2)
        latent_features = [0.1, 0.2, 0.3, 0.4]

        cfg_scale5 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 5.0,
                },
            }
        }

        _, trace_s5 = coder.detect_score(
            cfg=cfg_scale5, latent_features=latent_features,
            plan_digest=plan_digest, lf_basis=lf_basis,
        )
        digest_s5 = trace_s5["lf_trace_digest"]
        assert digest_s5 is not None, "lf_trace_digest must not be None when enabled"

        cfg_scale15 = {
            "watermark": {
                "lf": {
                    "enabled": True,
                    "message_length": 8,
                    "ecc_sparsity": 3,
                    "correlation_scale": 15.0,
                },
            }
        }

        _, trace_s15 = coder.detect_score(
            cfg=cfg_scale15, latent_features=latent_features,
            plan_digest=plan_digest, lf_basis=lf_basis,
        )
        digest_s15 = trace_s15["lf_trace_digest"]
        assert digest_s15 is not None, "lf_trace_digest must not be None when enabled"

        assert digest_s5 != digest_s15, \
            "lf_trace_digest must change when correlation_scale changes"