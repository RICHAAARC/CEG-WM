"""
File purpose: Paper faithfulness gate tests.
Module type: General module
"""

import pytest
from main.diffusion.sd3 import pipeline_inspector
from main.diffusion.sd3 import diffusion_tracer
from main.watermarking.paper_faithfulness import injection_site_binder
from main.watermarking.paper_faithfulness import alignment_evaluator


def test_pipeline_inspector_requires_non_none_pipeline() -> None:
    """
    功能：测试 pipeline_inspector 拒绝 None 输入。

    Test pipeline inspector rejects None pipeline input.
    """
    with pytest.raises(TypeError, match="pipeline_obj must not be None"):
        pipeline_inspector.inspect_sd3_pipeline(pipeline_obj=None)


def test_diffusion_tracer_init_requires_dict_cfg() -> None:
    """
    功能：测试 diffusion_tracer 初始化要求 dict cfg。

    Test diffusion tracer init requires dict cfg.
    """
    with pytest.raises(TypeError, match="cfg must be dict"):
        diffusion_tracer.init_tracer(cfg="not_a_dict")  # type: ignore


def test_diffusion_tracer_disabled_returns_absent() -> None:
    """
    功能：测试 tracer 禁用时返回 absent 语义。

    Test tracer returns absent semantics when disabled.
    """
    cfg = {
        "trajectory_sample_stride": 5,
        "num_inference_steps": 10
    }
    tracer_state = diffusion_tracer.init_tracer(cfg, enable_tracing=False)
    trajectory_evidence, spec_digest, traj_digest = diffusion_tracer.finalize_trajectory(tracer_state)

    assert trajectory_evidence["status"] == "absent"
    assert trajectory_evidence["trajectory_absent_reason"] == "tracing_disabled"
    assert spec_digest == "<absent>"
    assert traj_digest == "<absent>"


def test_injection_site_binder_requires_non_empty_hook_type() -> None:
    """
    功能：测试 injection_site_binder 要求非空 hook_type。

    Test injection site binder requires non-empty hook_type.
    """
    with pytest.raises(TypeError, match="hook_type must be non-empty string"):
        injection_site_binder.build_injection_site_spec(hook_type="")


def test_injection_site_digest_is_reproducible() -> None:
    """
    功能：测试 injection_site_digest 可复算。

    Test injection site digest is reproducible.
    """
    spec_1, digest_1 = injection_site_binder.build_injection_site_spec(
        hook_type="callback_on_step_end",
        target_tensor_name="latents",
        hook_timing="post"
    )

    spec_2, digest_2 = injection_site_binder.build_injection_site_spec(
        hook_type="callback_on_step_end",
        target_tensor_name="latents",
        hook_timing="post"
    )

    assert digest_1 == digest_2
    assert injection_site_binder.validate_injection_site_digest(spec_1, digest_1)


def test_alignment_evaluator_requires_dict_inputs() -> None:
    """
    功能：测试 alignment_evaluator 要求 dict 输入。

    Test alignment evaluator requires dict inputs.
    """
    with pytest.raises(TypeError, match="paper_spec must be dict"):
        alignment_evaluator.evaluate_alignment(
            paper_spec="not_a_dict",  # type: ignore
            pipeline_fingerprint=None,
            trajectory_evidence=None,
            injection_site_spec=None,
            cfg={}
        )

    with pytest.raises(TypeError, match="cfg must be dict"):
        alignment_evaluator.evaluate_alignment(
            paper_spec={},
            pipeline_fingerprint=None,
            trajectory_evidence=None,
            injection_site_spec=None,
            cfg="not_a_dict"  # type: ignore
        )


def test_alignment_evaluator_fails_if_pipeline_fingerprint_missing() -> None:
    """
    功能：测试缺少 pipeline_fingerprint 时对齐检查失败。

    Test alignment check fails if pipeline_fingerprint is missing.
    """
    paper_spec = {
        "sd3_adaptation": {
            "injection_site_binding": {
                "required_fields": []
            }
        },
        "alignment_check_rules": {
            "method_specific_parameter_binding": {
                "bindings": {}
            }
        }
    }

    alignment_report, _ = alignment_evaluator.evaluate_alignment(
        paper_spec=paper_spec,
        pipeline_fingerprint=None,
        trajectory_evidence=None,
        injection_site_spec=None,
        cfg={}
    )

    assert alignment_report["overall_status"] == "FAIL"
    assert alignment_report["fail_count"] > 0

    # 检查具体失败项。
    checks = alignment_report["checks"]
    pipeline_check = [c for c in checks if c["check_name"] == "pipeline_fingerprint_presence"][0]
    assert pipeline_check["result"] == "FAIL"


def test_alignment_evaluator_fails_if_injection_site_absent() -> None:
    """
    功能：测试 injection_site_spec 为 absent 时对齐检查失败。

    Test alignment check fails if injection_site_spec is absent.
    """
    paper_spec = {
        "sd3_adaptation": {
            "injection_site_binding": {
                "required_fields": ["hook_type", "target_tensor_name"]
            }
        },
        "alignment_check_rules": {
            "method_specific_parameter_binding": {
                "bindings": {}
            }
        }
    }

    pipeline_fingerprint = {
        "transformer_num_blocks": 24,
        "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
        "vae_latent_channels": 16
    }

    injection_site_spec = {
        "hook_type": "<absent>",
        "status": "absent"
    }

    alignment_report, _ = alignment_evaluator.evaluate_alignment(
        paper_spec=paper_spec,
        pipeline_fingerprint=pipeline_fingerprint,
        trajectory_evidence=None,
        injection_site_spec=injection_site_spec,
        cfg={}
    )

    assert alignment_report["overall_status"] == "FAIL"

    # 检查 injection_site_alignment 失败。
    checks = alignment_report["checks"]
    injection_check = [c for c in checks if c["check_name"] == "injection_site_alignment"][0]
    assert injection_check["result"] == "FAIL"


def test_alignment_evaluator_passes_with_complete_evidence() -> None:
    """
    功能：测试完整证据时对齐检查通过。

    Test alignment check passes with complete evidence.
    """
    paper_spec = {
        "sd3_adaptation": {
            "injection_site_binding": {
                "required_fields": ["hook_type"]
            }
        },
        "alignment_check_rules": {
            "method_specific_parameter_binding": {
                "bindings": {}
            }
        }
    }

    pipeline_fingerprint = {
        "transformer_num_blocks": 24,
        "scheduler_class_name": "FlowMatchEulerDiscreteScheduler",
        "vae_latent_channels": 16
    }

    trajectory_evidence = {
        "status": "ok",
        "trajectory_spec_digest": "a" * 64,
        "trajectory_digest": "b" * 64
    }

    injection_site_spec = {
        "hook_type": "callback_on_step_end",
        "target_tensor_name": "latents"
    }

    alignment_report, _ = alignment_evaluator.evaluate_alignment(
        paper_spec=paper_spec,
        pipeline_fingerprint=pipeline_fingerprint,
        trajectory_evidence=trajectory_evidence,
        injection_site_spec=injection_site_spec,
        cfg={}
    )

    assert alignment_report["overall_status"] == "PASS"
    assert alignment_report["fail_count"] == 0


def test_paper_spec_digest_mismatch_must_fail() -> None:
    """
    功能：测试未绑定 paper_spec 或 digest 不一致必须 FAIL。

    Test that missing or mismatched paper_spec_digest must FAIL.
    Regression test for S-D requirement.
    """
    from main.watermarking.detect import orchestrator

    # 模拟 input_record 缺少 paper_spec_digest
    input_record_missing = {
        "content_evidence": {
            "plan_digest": "test_plan_digest"
        }
    }

    status_missing, absent_reasons_missing, mismatch_reasons_missing, fail_reasons_missing = orchestrator._evaluate_paper_faithfulness_consistency(
        input_record=input_record_missing
    )

    assert status_missing == "mismatch"
    assert "paper_faithfulness_section_absent" in mismatch_reasons_missing

    # 模拟 input_record 的 paper_spec_digest 为 invalid
    input_record_invalid = {
        "paper_faithfulness": {
            "spec_digest": "<absent>"
        },
        "content_evidence": {
            "plan_digest": "test_plan_digest"
        }
    }

    status_invalid, absent_reasons_invalid, mismatch_reasons_invalid, fail_reasons_invalid = orchestrator._evaluate_paper_faithfulness_consistency(
        input_record=input_record_invalid
    )

    assert status_invalid == "mismatch"
    assert "paper_spec_digest_absent_or_invalid" in mismatch_reasons_invalid


def test_injection_site_change_must_update_digest() -> None:
    """
    功能：测试注入点变化但 injection_site_digest 未变化必须 FAIL。

    Test that injection site changes must be reflected in digest.
    Regression test for S-D requirement.
    """
    spec_1, digest_1 = injection_site_binder.build_injection_site_spec(
        hook_type="callback_on_step_end",
        target_module_name="transformer.blocks[0]",
        target_tensor_name="latents",
        hook_timing="post"
    )

    spec_2, digest_2 = injection_site_binder.build_injection_site_spec(
        hook_type="callback_on_step_end",
        target_module_name="transformer.blocks[1]",  # 变化：不同的 block
        target_tensor_name="latents",
        hook_timing="post"
    )

    # 不同的注入点必须产生不同的 digest
    assert digest_1 != digest_2


def test_pipeline_structure_change_must_update_fingerprint() -> None:
    """
    功能：测试 SD3 pipeline 结构变化必须更新 pipeline_fingerprint_digest。

    Test that SD3 pipeline structure changes must update fingerprint digest.
    Regression test for S-D requirement.
    """
    # 模拟两个不同结构的 pipeline config
    class MockPipeline1:
        class Transformer:
            class Config:
                num_layers = 24
                attention_head_dim = 64

            config = Config()

        transformer = Transformer()

    class MockPipeline2:
        class Transformer:
            class Config:
                num_layers = 32  # 变化：不同的层数
                attention_head_dim = 64

            config = Config()

        transformer = Transformer()

    fingerprint_1, digest_1 = pipeline_inspector.inspect_sd3_pipeline(MockPipeline1(), cfg={})
    fingerprint_2, digest_2 = pipeline_inspector.inspect_sd3_pipeline(MockPipeline2(), cfg={})

    # 不同的 pipeline 结构必须产生不同的 fingerprint digest
    assert digest_1 != digest_2
    assert fingerprint_1["transformer_num_blocks"] != fingerprint_2["transformer_num_blocks"]
