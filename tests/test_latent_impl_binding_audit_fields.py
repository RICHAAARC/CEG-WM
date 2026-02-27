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


def test_injection_cfg_marks_adapter_fallback_when_impl_not_channel_impl() -> None:
    cfg = {
        "impl": {
            "lf_coder_id": "lf_coder_prc_v1",
            "hf_embedder_id": "hf_embedder_t2smark_v1",
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
    assert lf_binding.get("impl_selected") == "lf_coder_prc_v1"
    assert hf_binding.get("impl_selected") == "hf_embedder_t2smark_v1"
    assert lf_binding.get("fallback_used") is True
    assert hf_binding.get("fallback_used") is True
    assert lf_binding.get("evidence_level") == "adapter_fallback"
    assert hf_binding.get("evidence_level") == "adapter_fallback"
    assert isinstance(lf_binding.get("fallback_reason"), str)
    assert isinstance(hf_binding.get("fallback_reason"), str)


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
    assert lf_binding.get("fallback_used") is False
    assert hf_binding.get("fallback_used") is False
    assert lf_binding.get("evidence_level") == "primary"
    assert hf_binding.get("evidence_level") == "primary"
    assert lf_binding.get("fallback_reason") is None
    assert hf_binding.get("fallback_reason") is None


def test_latent_step_embed_trace_contains_impl_binding_fields() -> None:
    cfg = {"paper_faithfulness": {"enabled": False}}
    injection_evidence = {
        "status": "ok",
        "injection_trace_digest": "a" * 64,
        "injection_params_digest": "b" * 64,
        "lf_impl_binding": {
            "impl_selected": "lf_coder_prc_v1",
            "adapter_path": "latent_modifier_channel_lf_v1",
            "fallback_used": True,
            "fallback_reason": "lf_impl_embed_interface_absent_for_latent_modifier",
        },
        "hf_impl_binding": {
            "impl_selected": "hf_embedder_t2smark_v1",
            "adapter_path": "latent_modifier_channel_hf_v1",
            "fallback_used": True,
            "fallback_reason": "hf_impl_embed_interface_absent_for_latent_modifier",
        },
    }
    trace = embed_orchestrator._build_latent_step_embed_trace(cfg, injection_evidence)
    assert isinstance(trace.get("lf_impl_binding"), dict)
    assert isinstance(trace.get("hf_impl_binding"), dict)
