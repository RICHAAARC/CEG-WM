"""
File purpose: latent per-step HF/LF 实现绑定审计字段回归测试。
Module type: General module
"""

from __future__ import annotations

from main.diffusion.sd3 import infer_runtime
from main.diffusion.sd3.callback_composer import InjectionContext
from main.watermarking.content_chain import channel_lf, channel_hf
from main.watermarking.embed import orchestrator as embed_orchestrator


def _build_context() -> InjectionContext:
    return InjectionContext(
        plan_digest="p" * 64,
        plan_ref={"rank": 4},
        lf_params_digest="l" * 64,
        hf_params_digest="h" * 64,
        enable_lf=True,
        enable_hf=True,
    )


def test_injection_cfg_marks_equivalent_binding_for_lf_hf_template_impls() -> None:
    cfg = {
        "impl": {
            "lf_coder_id": "low_freq_template_codec_v2",
            "hf_embedder_id": "high_freq_template_codec_v2",
        },
        "watermark": {
            "lf": {"strength": 1.5},
            "hf": {"threshold_percentile": 75.0},
        },
        "seed": 42,
    }
    context = _build_context()
    injection_cfg = infer_runtime._build_injection_cfg(cfg, context)

    lf_binding = injection_cfg.get("lf_impl_binding")
    hf_binding = injection_cfg.get("hf_impl_binding")
    assert isinstance(lf_binding, dict)
    assert isinstance(hf_binding, dict)
    assert lf_binding.get("impl_selected") == "low_freq_template_codec_v2"
    assert hf_binding.get("impl_selected") == "high_freq_template_codec_v2"
    # v2.0 收口： evidence_level 是正式路径的唯一分类字段；不写出 fallback_used/adapter_path。
    # v2 合并后 LF_CHANNEL_IMPL_ID == LOW_FREQ_TEMPLATE_CODEC_V2_ID，模板 codec v2 直接是 primary
    assert lf_binding.get("evidence_level") == "primary"
    assert hf_binding.get("evidence_level") == "primary"
    assert lf_binding.get("equivalence_mode") == "lf_template_to_channel_lf_parameter_mapping_v2"
    assert hf_binding.get("equivalence_mode") == "hf_template_to_channel_hf_parameter_mapping_v2"
    assert "fallback_used" not in lf_binding
    assert "fallback_used" not in hf_binding
    assert "adapter_path" not in lf_binding
    assert "adapter_path" not in hf_binding


def test_injection_cfg_marks_no_fallback_for_channel_native_impl() -> None:
    cfg = {
        "impl": {
            "lf_coder_id": channel_lf.LF_CHANNEL_IMPL_ID,
            "hf_embedder_id": channel_hf.HF_CHANNEL_IMPL_ID,
        },
        "watermark": {
            "lf": {"strength": 1.5},
            "hf": {"threshold_percentile": 75.0},
        },
    }
    context = _build_context()
    injection_cfg = infer_runtime._build_injection_cfg(cfg, context)

    lf_binding = injection_cfg.get("lf_impl_binding")
    hf_binding = injection_cfg.get("hf_impl_binding")
    assert isinstance(lf_binding, dict)
    assert isinstance(hf_binding, dict)
    # v2.0 收口： evidence_level 是正式路径的唯一分类字段；不写出 fallback_used/adapter_path。
    assert lf_binding.get("evidence_level") == "primary"
    assert hf_binding.get("evidence_level") == "primary"
    assert "fallback_used" not in lf_binding
    assert "fallback_used" not in hf_binding
    assert "adapter_path" not in lf_binding
    assert "adapter_path" not in hf_binding


def test_latent_step_embed_trace_contains_impl_binding_fields() -> None:
    cfg = {"paper_faithfulness": {"enabled": False}}
    injection_evidence = {
        "status": "ok",
        "injection_trace_digest": "a" * 64,
        "injection_params_digest": "b" * 64,
        "lf_impl_binding": {
            "impl_selected": "low_freq_template_codec_v2",
            "evidence_level": "adapter_fallback",
        },
        "hf_impl_binding": {
            "impl_selected": "high_freq_template_codec_v2",
            "evidence_level": "adapter_fallback",
        },
    }
    trace = embed_orchestrator._build_latent_step_embed_trace(cfg, injection_evidence)
    assert isinstance(trace.get("lf_impl_binding"), dict)
    assert isinstance(trace.get("hf_impl_binding"), dict)
