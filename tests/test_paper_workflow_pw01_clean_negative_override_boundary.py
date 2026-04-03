"""
File purpose: Freeze the override-first planner boundary for PW01 clean_negative detect.
Module type: General module
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast

import pytest

import main.watermarking.detect.orchestrator as detect_orchestrator_module
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision


class _ContentExtractorStub:
    def extract(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "ok",
            "score": 0.12,
            "content_chain_score": 0.12,
            "lf_score": 0.12,
            "hf_score": None,
            "score_parts": {},
            "content_failure_reason": None,
            "audit": {
                "impl_identity": "content_extractor_stub",
                "impl_version": "v1",
                "impl_digest": "content-extractor-stub-digest",
                "trace_digest": "content-extractor-stub-trace",
            },
        }


class _UnusedGeometryExtractor:
    def extract(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("geometry extractor should be bypassed by ablation gating")


class _UnusedSyncModule:
    def sync(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("sync module should not run when geometry is disabled")


class _FusionRuleStub:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def fuse(
        self,
        cfg: Dict[str, Any],
        content_evidence: Dict[str, Any],
        geometry_evidence: Dict[str, Any],
    ) -> FusionDecision:
        self.calls.append(
            {
                "cfg": dict(cfg),
                "content_evidence": dict(content_evidence),
                "geometry_evidence": dict(geometry_evidence),
            }
        )
        return FusionDecision(
            is_watermarked=False,
            decision_status="decided",
            thresholds_digest="thresholds-digest-stub",
            evidence_summary={
                "content_score": content_evidence.get("content_score") or content_evidence.get("score"),
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": "fusion_rule_stub",
            },
            audit={"impl_identity": "fusion_rule_stub"},
            fusion_rule_version="v1",
        )


class _CountingSubspacePlanner:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: Any = None,
        cfg_digest: str | None = None,
        inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "cfg": dict(cfg),
                "mask_digest": mask_digest,
                "cfg_digest": cfg_digest,
                "inputs": dict(inputs) if isinstance(inputs, dict) else None,
            }
        )
        return {
            "status": "ok",
            "plan_digest": "plan-probe",
            "basis_digest": "basis-probe",
            "planner_input_digest": "plan-input-probe",
            "plan": {
                "planner_input_digest": "plan-input-probe",
                "rank": 2,
                "planner_impl_identity": {
                    "impl_id": "planner_impl",
                    "impl_version": "v1",
                    "impl_digest": "planner-digest",
                },
            },
            "plan_failure_reason": None,
        }

    def _build_planner_input_digest(self, inputs: Dict[str, Any]) -> str:
        _ = inputs
        return "plan-input-probe"


def _build_clean_negative_style_input_record() -> Dict[str, Any]:
    """
    Build a clean_negative-style input record with probe-derived planner truth.

    Args:
        None.

    Returns:
        Input record mapping.
    """
    planner_identity = {
        "impl_id": "planner_impl",
        "impl_version": "v1",
        "impl_digest": "planner-digest",
    }
    return {
        "operation": "embed_preview_input",
        "event_id": "evt_clean_negative_boundary",
        "watermarked_path": "preview.png",
        "image_path": "preview.png",
        "prompt": "boundary prompt",
        "prompt_text": "boundary prompt",
        "prompt_sha256": "prompt-sha",
        "seed": 7,
        "plan_digest": "plan-probe",
        "basis_digest": "basis-probe",
        "plan_input_digest": "plan-input-probe",
        "plan_input_schema_version": "v2",
        "subspace_planner_impl_identity": dict(planner_identity),
        "subspace_plan": {
            "planner_input_digest": "plan-input-probe",
            "rank": 2,
            "planner_impl_identity": dict(planner_identity),
        },
        "negative_branch_source_attestation_provenance": {
            "statement": {
                "schema": "gen_attest_v1",
                "model_id": "stub-model",
                "prompt_commit": "prompt-commit",
                "seed_commit": "seed-commit",
                "plan_digest": "plan-probe",
                "event_nonce": "nonce",
                "time_bucket": "2026-01-01",
            },
            "attestation_digest": "attestation-digest",
        },
    }


def _build_detect_cfg() -> Dict[str, Any]:
    """
    Build a minimal detect cfg for override-boundary tests.

    Args:
        None.

    Returns:
        Runtime config mapping.
    """
    return {
        "policy_path": "content_np_geo_rescue",
        "paper_faithfulness": {"enabled": False},
        "attestation": {"enabled": True, "use_trajectory_mix": False},
        "ablation": {
            "normalized": {
                "enable_content": True,
                "enable_geometry": False,
                "enable_sync": False,
                "enable_anchor": False,
                "enable_image_sidecar": False,
            }
        },
        "watermark": {
            "hf": {"enabled": False},
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 4,
                "feature_dim": 8,
            },
        },
        "inference_num_steps": 28,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
    }


def _build_content_result_override() -> Dict[str, Any]:
    """
    Build a stable detect-side content result override.

    Args:
        None.

    Returns:
        Content evidence payload mapping.
    """
    return {
        "status": "ok",
        "score": 0.12,
        "content_chain_score": 0.12,
        "mask_digest": "mask-digest-stub",
        "audit": {
            "impl_identity": "content_override_stub",
            "impl_version": "v1",
            "impl_digest": "content-override-digest",
            "trace_digest": "content-override-trace",
        },
    }


def _build_impl_set(planner: _CountingSubspacePlanner) -> SimpleNamespace:
    """
    Build the minimal impl_set required by run_detect_orchestrator.

    Args:
        planner: Counting planner stub.

    Returns:
        Namespace with the required detect implementations.
    """
    return SimpleNamespace(
        content_extractor=_ContentExtractorStub(),
        geometry_extractor=_UnusedGeometryExtractor(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=planner,
        sync_module=_UnusedSyncModule(),
        hf_embedder=None,
    )


def _patch_attestation_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch detect attestation preparation with stable runtime bindings.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        detect_orchestrator_module,
        "_prepare_detect_attestation_context",
        lambda cfg, input_record: {
            "attestation_status": "ok",
            "attestation_source": "negative_branch_statement_only_provenance",
            "runtime_bindings": {
                "attestation_digest": "attestation-digest",
                "event_binding_digest": "event-binding-digest",
                "k_lf": "lf-key",
                "k_hf": "hf-key",
                "k_geo": "geo-key",
                "geo_anchor_seed": 17,
                "event_binding_mode": "statement_only",
            },
        },
    )
    monkeypatch.setattr(
        detect_orchestrator_module,
        "_build_hf_detect_evidence",
        lambda **kwargs: {
            "status": "absent",
            "hf_score": None,
            "hf_trace_digest": None,
            "hf_evidence_summary": {
                "hf_status": "absent",
                "hf_absent_reason": "formal_profile_sidecar_disabled",
            },
            "content_failure_reason": "formal_profile_sidecar_disabled",
        },
    )
    monkeypatch.setattr(
        detect_orchestrator_module,
        "_extract_lf_raw_score_from_trajectory",
        lambda **kwargs: (
            0.12,
            {
                "lf_status": "ok",
                "lf_trace_digest": "lf-trace-digest-stub",
                "bp_converged": True,
                "bp_iteration_count": 3,
                "parity_check_digest": "parity-check-digest-stub",
                "lf_detect_path": "low_freq_template_trajectory",
            },
        ),
    )


def test_clean_negative_detect_override_short_circuits_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Freeze the current override-first detect boundary for clean_negative.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Notes:
        This test freezes a current architecture fact rather than a general law:
        once run_detect has already materialized detect_plan_result_override from
        input_record.subspace_plan, detect-side attestation bindings may still be
        injected into LF/HF/GEO runtime, but planner selection itself must remain
        short-circuited. If a future refactor stops using override-first, this
        boundary must be re-audited instead of silently reinterpreted.
    """
    _patch_attestation_runtime(monkeypatch)
    planner = _CountingSubspacePlanner()
    input_record = _build_clean_negative_style_input_record()
    cfg = _build_detect_cfg()

    detect_plan_result_override = {
        "status": "ok",
        "plan_digest": input_record["plan_digest"],
        "basis_digest": input_record["basis_digest"],
        "planner_input_digest": input_record["plan_input_digest"],
        "plan": cast(Dict[str, Any], input_record["subspace_plan"]),
        "plan_failure_reason": None,
    }

    record = run_detect_orchestrator(
        cfg=cfg,
        impl_set=_build_impl_set(planner),
        input_record=input_record,
        cfg_digest="cfg-digest-stub",
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=_build_content_result_override(),
        detect_plan_result_override=detect_plan_result_override,
    )

    assert planner.calls == []
    assert cfg["k_lf"] == "lf-key"
    assert cfg["k_hf"] == "hf-key"
    assert cfg["k_geo"] == "geo-key"
    assert cfg["geo_anchor_seed"] == 17
    assert record["subspace_plan"] == input_record["subspace_plan"]
    assert record["subspace_planner_impl_identity"] == input_record["subspace_planner_impl_identity"]
    assert record["plan_input_digest"] == input_record["plan_input_digest"]
    assert record["plan_input_schema_version"] == input_record["plan_input_schema_version"]
    assert record["plan_digest"] == input_record["plan_digest"]
    assert record["basis_digest"] == input_record["basis_digest"]
    assert record["plan_digest_expected"] == input_record["plan_digest"]
    assert record["plan_digest_validation_status"] == "ok"


def test_clean_negative_detect_replans_when_override_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Freeze the control branch for the current override-first architecture.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Notes:
        This control test exists to keep the previous boundary honest. The
        current clean_negative path is safe from detect-side replanning only
        because override-first remains in place; if override disappears, detect
        falls back to planner execution and the planner path becomes audit-relevant.
    """
    _patch_attestation_runtime(monkeypatch)
    planner = _CountingSubspacePlanner()
    input_record = _build_clean_negative_style_input_record()

    record = run_detect_orchestrator(
        cfg=_build_detect_cfg(),
        impl_set=_build_impl_set(planner),
        input_record=input_record,
        cfg_digest="cfg-digest-stub",
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=_build_content_result_override(),
        detect_plan_result_override=None,
    )

    assert len(planner.calls) == 1
    planner_call = planner.calls[0]
    assert planner_call["mask_digest"] == "mask-digest-stub"
    assert isinstance(planner_call["inputs"], dict)
    assert cast(Dict[str, Any], planner_call["inputs"])["trace_signature"]["num_inference_steps"] == 28
    assert record["subspace_plan"]["planner_input_digest"] == "plan-input-probe"
    assert record["subspace_planner_impl_identity"] == {
        "impl_id": "planner_impl",
        "impl_version": "v1",
        "impl_digest": "planner-digest",
    }
    assert record["plan_input_digest"] == "plan-input-probe"
    assert record["plan_input_schema_version"] == "v2"
    assert record["plan_digest"] == "plan-probe"
    assert record["basis_digest"] == "basis-probe"
    assert record["plan_digest_expected"] == input_record["plan_digest"]
    assert record["plan_digest_validation_status"] == "ok"