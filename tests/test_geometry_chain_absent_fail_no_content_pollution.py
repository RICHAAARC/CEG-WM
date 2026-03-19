"""
File purpose: 验证几何链 absent/fail 语义不会污染主链得分与判决输入。
Module type: General module
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision


class _ContentExtractorStub:
    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        cfg_digest: str | None = None,
    ) -> ContentEvidence:
        _ = inputs
        _ = cfg_digest
        return ContentEvidence(
            status="ok",
            score=0.37,
            audit={
                "impl_identity": "content_stub",
                "impl_version": "v1",
                "impl_digest": "b" * 64,
                "trace_digest": "c" * 64,
            },
        )


class _GeometryExtractorStub:
    def __init__(self, status: str, reason: str | None) -> None:
        self._status = status
        self._reason = reason

    def extract(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": self._status,
            "geo_score": None,
            "geo_failure_reason": self._reason,
            "audit": {
                "impl_identity": "geometry_stub",
                "impl_version": "v1",
                "impl_digest": "d" * 64,
                "trace_digest": "e" * 64,
                "sync_status_detail": self._status,
            },
        }


class _FusionRuleStub:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
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
    def planner_input_digest(self) -> str:
        return "planner_input_ok"

    @property
    def plan(self) -> Dict[str, Any]:
        return {
            "planner_input_digest": "planner_input_ok",
            "planner_impl_identity": {
                "impl_id": "planner_stub",
                "impl_version": "v1",
                "impl_digest": "f" * 64,
            }
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "basis_digest": self.basis_digest,
            "plan": self.plan,
        }


class _SubspacePlannerStub:
    def plan(self, cfg: Dict[str, Any], mask_digest: str | None = None, cfg_digest: str | None = None, inputs: Dict[str, Any] | None = None) -> _PlanResult:
        _ = mask_digest
        _ = cfg_digest
        _ = inputs
        return _PlanResult(plan_digest="plan_ok", basis_digest="basis_ok")


class _SyncStub:
    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "ok"}


def _build_cfg() -> Dict[str, Any]:
    return {
        "evaluate": {"target_fpr": 0.01},
        "inference_num_steps": 12,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
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
        "ablation": {
            "normalized": {
                "enable_content": True,
                "enable_geometry": True,
                "enable_fusion": True,
                "enable_mask": True,
                "enable_subspace": True,
                "enable_rescue": False,
                "enable_lf": True,
                "enable_hf": False,
                "lf_only": False,
                "hf_only": False,
            }
        },
    }


def _run_with_geometry_status(status: str, reason: str | None) -> Dict[str, Any]:
    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(),
        geometry_extractor=_GeometryExtractorStub(status=status, reason=reason),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_SubspacePlannerStub(),
        sync_module=_SyncStub(),
    )
    input_record = {
        "plan_digest": "plan_ok",
        "basis_digest": "basis_ok",
        "subspace_plan": {
            "planner_input_digest": "planner_input_ok",
        },
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
    detect_plan_result = _PlanResult(plan_digest="plan_ok", basis_digest="basis_ok")
    return run_detect_orchestrator(
        _build_cfg(),
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
        detect_plan_result_override=detect_plan_result,
    )


def test_geometry_absent_or_fail_does_not_change_content_score() -> None:
    """
    功能：几何链 absent/fail 不得改变主链 content_score 与判决状态。

    Geometry absent/fail must not alter content score or decision status.

    Args:
        None.

    Returns:
        None.
    """
    absent_record = _run_with_geometry_status("absent", "anchor_disabled_by_policy")
    fail_record = _run_with_geometry_status("fail", "attention_anchor_extraction_failed")

    absent_fusion = absent_record["fusion_result"]
    fail_fusion = fail_record["fusion_result"]

    assert absent_fusion.evidence_summary.get("content_score") == 0.37
    assert fail_fusion.evidence_summary.get("content_score") == 0.37
    assert absent_fusion.decision_status == "decided"
    assert fail_fusion.decision_status == "decided"
    assert absent_record["execution_report"]["geometry_chain_status"] == "absent"
    assert fail_record["execution_report"]["geometry_chain_status"] == "failed"
    assert absent_record["execution_report"]["fusion_status"] == "ok"
    assert fail_record["execution_report"]["fusion_status"] == "ok"
