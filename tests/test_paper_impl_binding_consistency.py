"""
File purpose: paper 妯″紡 impl 缁戝畾涓€鑷存€ч棬绂佸洖褰掓祴璇曘€?Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

from main.watermarking.detect import orchestrator as detect_orchestrator


def test_paper_impl_binding_rejects_fallback_binding() -> None:
    cfg: Dict[str, Any] = {"paper_faithfulness": {"enabled": True}}
    injection_evidence: Dict[str, Any] = {
        "lf_impl_binding": {
            "impl_selected": "low_freq_template_codec_v2",
            "evidence_level": "non_compliant",
        },
        "hf_impl_binding": {
            "impl_selected": "high_freq_template_codec_v2",
            "evidence_level": "primary",
        },
    }

    evaluate_consistency = getattr(detect_orchestrator, "_evaluate_paper_impl_binding_consistency")
    status, reason = evaluate_consistency(cfg, injection_evidence)
    assert status == "mismatch"
    assert reason == "lf_impl_binding_non_primary_binding_under_paper_mode"


def test_paper_impl_binding_accepts_primary_bindings() -> None:
    cfg: Dict[str, Any] = {"paper_faithfulness": {"enabled": True}}
    injection_evidence: Dict[str, Any] = {
        "lf_impl_binding": {
            "impl_selected": "channel_lf_v1",
            "evidence_level": "primary",
        },
        "hf_impl_binding": {
            "impl_selected": "channel_hf_v1",
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
    reason, field_path = resolve_primary(["lf_impl_binding_non_primary_binding_under_paper_mode"])
    assert reason == "lf_impl_binding_non_primary_binding_under_paper_mode"
    assert field_path == "content_evidence.lf_impl_binding.evidence_level"


def test_paper_impl_binding_rejects_int_ecc_under_paper_mode() -> None:
    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "lf": {
                "enabled": True,
                "ecc": 3,
            }
        },
    }
    injection_evidence: Dict[str, Any] = {
        "status": "ok",
        "lf_impl_binding": {
            "impl_selected": "low_freq_template_codec_v2",
            "evidence_level": "primary",
        },
        "hf_impl_binding": {
            "impl_selected": "channel_hf_v1",
            "evidence_level": "primary",
        },
    }

    evaluate_consistency = getattr(detect_orchestrator, "_evaluate_paper_impl_binding_consistency")
    status, reason = evaluate_consistency(cfg, injection_evidence)
    assert status == "mismatch"
    assert reason == "lf_ecc_int_not_allowed_under_paper_mode"


def test_primary_mismatch_maps_lf_ecc_int_reason_to_field_path() -> None:
    resolve_primary = getattr(detect_orchestrator, "_resolve_primary_mismatch")
    reason, field_path = resolve_primary(["lf_ecc_int_not_allowed_under_paper_mode"])
    assert reason == "lf_ecc_int_not_allowed_under_paper_mode"
    assert field_path == "watermark.lf.ecc"

