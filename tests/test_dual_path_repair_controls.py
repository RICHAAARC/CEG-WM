"""
File purpose: dual-path repair 开关与 LF exact repair 辅助函数回归测试。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from main.cli import run_detect


def _build_embed_input_record() -> Dict[str, Any]:
    return {
        "content_evidence": {
            "injection_metrics": {
                "lf_closed_loop_summary": {
                    "pre_injection_coeffs": [-0.4, 0.2, -0.1],
                    "injected_template_coeffs": [0.3, -0.5, 0.4],
                    "post_injection_coeffs": [-0.1, -0.3, 0.3],
                    "expected_bit_signs": [1, -1, 1],
                },
                "lf_edit_timestep_closed_loop_summary": {
                    "post_injection_coeffs": [-0.2, -0.15, 0.25],
                },
                "lf_terminal_step_closed_loop_summary": {
                    "post_injection_coeffs": [-0.25, -0.1, 0.2],
                },
                "lf_closed_loop_digest": "a" * 64,
                "lf_closed_loop_step_index": 12,
                "lf_closed_loop_selection_rule": "max_lf_delta_norm",
                "lf_edit_timestep_step_index": 12,
                "lf_terminal_step_index": 15,
            }
        }
    }


def test_paper_profiles_keep_lf_on_and_geo_off_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for relative_path in [
        "configs/paper_full_cuda.yaml",
        "configs/paper_full_cuda_mini_real_validation.yaml",
    ]:
        cfg = yaml.safe_load((repo_root / relative_path).read_text(encoding="utf-8"))
        assert isinstance(cfg, dict)
        detect_cfg = cfg.get("detect") if isinstance(cfg.get("detect"), dict) else {}
        content_cfg = detect_cfg.get("content") if isinstance(detect_cfg.get("content"), dict) else {}
        geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
        lf_exact_repair_cfg = content_cfg.get("lf_exact_repair") if isinstance(content_cfg.get("lf_exact_repair"), dict) else {}
        geo_score_repair_cfg = geometry_cfg.get("geo_score_repair") if isinstance(geometry_cfg.get("geo_score_repair"), dict) else {}

        assert lf_exact_repair_cfg.get("enabled") is True
        assert lf_exact_repair_cfg.get("mode") == "host_template_recenter"
        assert geo_score_repair_cfg.get("enabled") is False
        assert geo_score_repair_cfg.get("mode") == "template_confidence"


def test_lf_exact_repair_recenter_can_be_enabled_or_disabled() -> None:
    trace_bundle = {
        "lf_attestation_features": [-0.2, -0.2, 0.1],
        "projected_lf_digest": "z" * 64,
    }
    embed_lf_context = run_detect._extract_embed_lf_closed_loop_context(_build_embed_input_record())  # pyright: ignore[reportPrivateUsage]

    repaired = run_detect._apply_lf_exact_repair_to_trace_bundle(  # pyright: ignore[reportPrivateUsage]
        {"detect": {"content": {"lf_exact_repair": {"enabled": True, "mode": "host_template_recenter"}}}},
        trace_bundle,
        embed_lf_context,
        repair_target="formal_exact",
    )
    disabled = run_detect._apply_lf_exact_repair_to_trace_bundle(  # pyright: ignore[reportPrivateUsage]
        {"detect": {"content": {"lf_exact_repair": {"enabled": False, "mode": "host_template_recenter"}}}},
        trace_bundle,
        embed_lf_context,
        repair_target="formal_exact",
    )

    assert repaired.get("lf_attestation_features") == pytest.approx([0.5, -0.9, 0.6])
    assert repaired.get("lf_exact_repair_applied") is True
    assert repaired.get("lf_exact_repair_summary", {}).get("status") == "applied"
    assert repaired.get("projected_lf_digest") != trace_bundle.get("projected_lf_digest")

    assert disabled.get("lf_attestation_features") == trace_bundle.get("lf_attestation_features")
    assert disabled.get("lf_exact_repair_applied") is False
    assert disabled.get("lf_exact_repair_summary", {}).get("status") == "disabled"


def test_build_lf_formal_exact_context_preserves_input_image_conditioned_source_when_repaired(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "detect.png"
    image_path.write_bytes(b"stub")

    def _fake_extract_image_conditioned_latent(*_args, **_kwargs):
        return {"status": "ok", "latent_array": [[[[0.0]]]]}

    def _fake_extract_trace_bundle(*_args, **_kwargs):
        return {
            "lf_attestation_features": [-0.2, -0.2, 0.1],
            "projected_lf_digest": "x" * 64,
        }

    monkeypatch.setattr(run_detect.infer_runtime, "extract_image_conditioned_latent", _fake_extract_image_conditioned_latent)
    monkeypatch.setattr(run_detect, "_extract_lf_attestation_trace_bundle_from_trajectory", _fake_extract_trace_bundle)
    monkeypatch.setattr(run_detect, "_resolve_detect_image_path_with_source", lambda *_args, **_kwargs: (image_path, "input_record.watermarked_path"))

    cfg = {
        "detect": {
            "content": {"lf_exact_repair": {"enabled": True, "mode": "host_template_recenter"}},
        }
    }
    plan_payload = {
        "plan": {
            "lf_basis": {
                "trajectory_feature_spec": {"edit_timestep": 0},
            }
        }
    }

    context = run_detect._build_lf_formal_exact_context(  # pyright: ignore[reportPrivateUsage]
        cfg,
        _build_embed_input_record(),
        plan_payload,
        pipeline_obj=object(),
        device="cpu",
    )

    assert context.get("formal_exact_evidence_source") == "input_image_conditioned_reconstruction"
    assert context.get("formal_exact_object_binding_status") == "ok"
    assert context.get("lf_exact_repair_applied") is True
    assert context.get("lf_exact_repair_summary", {}).get("status") == "applied"
    trace_bundle = context.get("formal_exact_trace_bundle")
    assert isinstance(trace_bundle, dict)
    assert trace_bundle.get("lf_attestation_features") == pytest.approx([0.5, -0.9, 0.6])