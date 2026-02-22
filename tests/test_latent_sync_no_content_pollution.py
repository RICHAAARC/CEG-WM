"""
File purpose: 验证 latent sync 不可用不会污染主链 content_score。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision
from main.watermarking.geometry_chain.sync.latent_sync_template import LatentSyncGeometryExtractor


class _ContentExtractorStub:
    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        cfg_digest: str | None = None,
    ) -> ContentEvidence:
        _ = cfg
        _ = inputs
        _ = cfg_digest
        return ContentEvidence(
            status="ok",
            score=0.55,
            audit={
                "impl_identity": "content_stub",
                "impl_version": "v1",
                "impl_digest": "b" * 64,
                "trace_digest": "c" * 64,
            },
        )


class _FusionRuleStub:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
        _ = cfg
        _ = geometry_evidence
        return FusionDecision(
            is_watermarked=False,
            decision_status="decided",
            thresholds_digest="threshold_stub",
            evidence_summary={
                "content_score": content_evidence.get("score"),
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": "fusion_stub",
            },
            audit={"impl": "fusion_stub"},
        )


@dataclass
class _PlanResult:
    plan_digest: str
    basis_digest: str

    @property
    def plan(self) -> Dict[str, Any]:
        return {
            "planner_input_digest": "planner_input_ok",
            "planner_impl_identity": {
                "impl_id": "planner_stub",
                "impl_version": "v1",
                "impl_digest": "f" * 64,
            },
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "basis_digest": self.basis_digest,
            "plan": self.plan,
        }


class _SubspacePlannerStub:
    def plan(self, cfg: Dict[str, Any], mask_digest: str | None = None, cfg_digest: str | None = None, inputs: Dict[str, Any] | None = None) -> _PlanResult:
        _ = cfg
        _ = mask_digest
        _ = cfg_digest
        _ = inputs
        return _PlanResult(plan_digest="plan_ok", basis_digest="basis_ok")


class _SyncStub:
    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "ok"}


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def _build_cfg(enable_latent_sync: bool) -> Dict[str, Any]:
    return {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "evaluate": {"target_fpr": 0.01},
        "inference_num_steps": 12,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_latent_sync": enable_latent_sync,
            }
        },
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 8,
                "feature_dim": 16,
                "seed": 3,
                "timestep_start": 0,
                "timestep_end": 7,
            }
        },
    }


def _run_detect_with_sync(enable_latent_sync: bool) -> Dict[str, Any]:
    geometry_extractor = LatentSyncGeometryExtractor("geometry_latent_sync_sd3_v1", "v1", "a" * 64)
    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(),
        geometry_extractor=geometry_extractor,
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_SubspacePlannerStub(),
        sync_module=_SyncStub(),
    )
    input_record: Dict[str, Any] = {
        "plan_digest": "plan_ok",
        "basis_digest": "basis_ok",
        "subspace_plan": {"planner_input_digest": "planner_input_ok"},
        "trajectory_evidence": {
            "status": "ok",
            "trajectory_spec_digest": "7" * 64,
            "trajectory_digest": "8" * 64,
        },
        "content_evidence": {
            "injection_status": "ok",
            "injection_trace_digest": "1" * 64,
            "injection_params_digest": "2" * 64,
            "subspace_binding_digest": "3" * 64,
        },
        "subspace_planner_impl_identity": {
            "impl_id": "planner_stub",
            "impl_version": "v1",
            "impl_digest": "f" * 64,
        },
    }
    cfg = _build_cfg(enable_latent_sync=enable_latent_sync)
    cfg["__detect_pipeline_obj__"] = _Pipeline()
    cfg["__detect_final_latents__"] = np.random.RandomState(6).randn(1, 4, 8, 8).astype(np.float32)
    return run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="cfg_digest_stub",
        trajectory_evidence={
            "status": "ok",
            "trajectory_spec_digest": "7" * 64,
            "trajectory_digest": "8" * 64,
        },
        injection_evidence={
            "status": "ok",
            "injection_trace_digest": "1" * 64,
            "injection_params_digest": "2" * 64,
            "subspace_binding_digest": "3" * 64,
        },
        detect_plan_result_override=_PlanResult(plan_digest="plan_ok", basis_digest="basis_ok"),
    )


def test_sync_absent_does_not_change_content_score() -> None:
    """
    功能：sync 不可用时不应污染主链 content_score。

    Sync absent must not alter content score semantics.

    Args:
        None.

    Returns:
        None.
    """
    record_disabled = _run_detect_with_sync(enable_latent_sync=False)
    record_enabled = _run_detect_with_sync(enable_latent_sync=True)

    fusion_disabled = record_disabled["fusion_result"]
    fusion_enabled = record_enabled["fusion_result"]

    assert fusion_disabled.evidence_summary.get("content_score") == 0.55
    assert fusion_enabled.evidence_summary.get("content_score") == 0.55
    assert fusion_disabled.decision_status == "decided"
    assert fusion_enabled.decision_status == "decided"
