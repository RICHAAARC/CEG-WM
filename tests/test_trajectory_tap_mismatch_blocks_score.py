"""
File purpose: trajectory tap mismatch 必须阻断 score 的回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict, cast

from main.core import digests
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.watermarking.content_chain.subspace.placeholder_planner import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
)
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision


class _ContentExtractorStub:
    def extract(self, cfg: Dict[str, Any]) -> ContentEvidence:
        return ContentEvidence(
            status="ok",
            score=0.9,
            audit={
                "impl_identity": "content_stub",
                "impl_version": "v1",
                "impl_digest": "digest",
                "trace_digest": "trace"
            },
            mask_digest="mask_digest_1"
        )


class _GeometryExtractorStub:
    def extract(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "absent", "geo_score": None}


class _FusionRuleStub:
    def __init__(self) -> None:
        self.called = False

    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
        self.called = True
        return FusionDecision(
            is_watermarked=False,
            decision_status="decided",
            thresholds_digest="stub",
            evidence_summary={
                "content_score": content_evidence.get("score"),
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": "fusion_stub"
            },
            audit={"impl": "fusion_stub"}
        )


class _SyncStub:
    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok"}


def _build_trace_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    generation_value = cfg.get("generation")
    model_value = cfg.get("model")
    generation: Dict[str, Any] = cast(Dict[str, Any], generation_value) if isinstance(generation_value, dict) else {}
    model: Dict[str, Any] = cast(Dict[str, Any], model_value) if isinstance(model_value, dict) else {}
    return {
        "num_inference_steps": cfg.get("inference_num_steps", generation.get("num_inference_steps", 16)),
        "guidance_scale": cfg.get("inference_guidance_scale", generation.get("guidance_scale", 7.0)),
        "height": cfg.get("inference_height", model.get("height", 512)),
        "width": cfg.get("inference_width", model.get("width", 512)),
    }


def test_trajectory_tap_mismatch_blocks_score() -> None:
    """
    功能：trajectory_digest 不一致必须触发 mismatch，且 content_score 为空。

    Trajectory digest mismatch must force mismatch and block content score.

    Args:
        None.

    Returns:
        None.
    """
    impl_digest = digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
    subspace_planner = SubspacePlannerImpl(SUBSPACE_PLANNER_ID, SUBSPACE_PLANNER_VERSION, impl_digest)
    fusion_stub = _FusionRuleStub()

    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=fusion_stub,
        subspace_planner=subspace_planner,
        sync_module=_SyncStub()
    )

    cfg: Dict[str, Any] = {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 10,
                "feature_dim": 24,
                "seed": 17,
                "timestep_start": 0,
                "timestep_end": 8
            }
        },
        "generation": {
            "num_inference_steps": 16,
            "guidance_scale": 7.5
        },
        "model": {
            "height": 512,
            "width": 512,
            "model_id": "sd3-test",
            "model_revision": "rev-1"
        },
        "evaluate": {
            "target_fpr": 1e-6
        }
    }

    detect_trajectory = {
        "status": "ok",
        "trajectory_spec_digest": "spec_digest_same",
        "trajectory_digest": "traj_digest_detect"
    }
    build_planner_input_digest = getattr(subspace_planner, "_build_planner_input_digest")
    planner_input_digest = build_planner_input_digest(
        {
            "trace_signature": _build_trace_signature(cfg),
            "trajectory_evidence": detect_trajectory
        }
    )

    input_record: Dict[str, Any] = {
        "content_evidence_payload": {
            "trajectory_evidence": {
                "status": "ok",
                "trajectory_spec_digest": "spec_digest_same",
                "trajectory_digest": "traj_digest_embed"
            }
        },
        "subspace_plan": {
            "verifiable_input_domain_spec": {
                "planner_input_digest": planner_input_digest
            }
        }
    }

    record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="cfg_digest_test",
        trajectory_evidence=detect_trajectory
    )

    assert record["content_result"]["status"] == "mismatch"
    assert record["content_result"]["score"] is None
    assert record["fusion_result"].decision_status == "error"
    assert record["fusion_result"].evidence_summary["content_score"] is None
    assert fusion_stub.called is False
