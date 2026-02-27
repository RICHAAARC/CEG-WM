"""
File purpose: paper 模式 impl 绑定一致性门禁回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.watermarking.detect import orchestrator as detect_orchestrator


def test_paper_impl_binding_rejects_fallback_binding() -> None:
    cfg: Dict[str, Any] = {"paper_faithfulness": {"enabled": True}}
    injection_evidence: Dict[str, Any] = {
        "lf_impl_binding": {
            "impl_selected": "lf_coder_prc_v1",
            "adapter_path": "latent_modifier_channel_lf_v1",
            "fallback_used": True,
            "fallback_reason": "lf_impl_embed_interface_absent_for_latent_modifier",
            "evidence_level": "adapter_fallback",
        },
        "hf_impl_binding": {
            "impl_selected": "hf_embedder_t2smark_v1",
            "adapter_path": "latent_modifier_channel_hf_v1",
            "fallback_used": False,
            "fallback_reason": None,
            "evidence_level": "primary",
        },
    }

    evaluate_consistency = getattr(detect_orchestrator, "_evaluate_paper_impl_binding_consistency")
    status, reason = evaluate_consistency(cfg, injection_evidence)
    assert status == "mismatch"
    assert reason == "lf_impl_binding_fallback_used_under_paper_mode"


def test_paper_impl_binding_accepts_primary_bindings() -> None:
    cfg: Dict[str, Any] = {"paper_faithfulness": {"enabled": True}}
    injection_evidence: Dict[str, Any] = {
        "lf_impl_binding": {
            "impl_selected": "channel_lf_v1",
            "adapter_path": "latent_modifier_channel_lf_v1",
            "fallback_used": False,
            "fallback_reason": None,
            "evidence_level": "primary",
        },
        "hf_impl_binding": {
            "impl_selected": "channel_hf_v1",
            "adapter_path": "latent_modifier_channel_hf_v1",
            "fallback_used": False,
            "fallback_reason": None,
            "evidence_level": "primary",
        },
    }

    evaluate_consistency = getattr(detect_orchestrator, "_evaluate_paper_impl_binding_consistency")
    status, reason = evaluate_consistency(cfg, injection_evidence)
    assert status == "ok"
    assert reason is None


def test_paper_impl_binding_non_ok_detect_evidence_returns_absent() -> None:
    cfg: Dict[str, Any] = {"paper_faithfulness": {"enabled": True}}
    injection_evidence: Dict[str, Any] = {
        "status": "absent",
        "injection_absent_reason": "inference_failed",
    }

    evaluate_consistency = getattr(detect_orchestrator, "_evaluate_paper_impl_binding_consistency")
    status, reason = evaluate_consistency(cfg, injection_evidence)
    assert status == "absent"
    assert reason == "paper_impl_binding_injection_status_not_ok"


def test_primary_mismatch_maps_paper_impl_binding_reason_to_field_path() -> None:
    resolve_primary = getattr(detect_orchestrator, "_resolve_primary_mismatch")
    reason, field_path = resolve_primary(["lf_impl_binding_fallback_used_under_paper_mode"])
    assert reason == "lf_impl_binding_fallback_used_under_paper_mode"
    assert field_path == "content_evidence.lf_impl_binding.fallback_used"
