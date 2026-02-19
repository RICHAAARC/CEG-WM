"""
File purpose: Detect-side mismatch semantics test for plan anchor inconsistency.
Module type: General module
"""

from dataclasses import dataclass
from typing import Any, Dict

from main.core import digests
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.watermarking.content_chain.subspace.placeholder_planner import (
    SubspacePlannerImpl,
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION
)
from main.watermarking.detect.orchestrator import run_detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision


class _ContentExtractorStub:
    def extract(self, cfg: Dict[str, Any]) -> ContentEvidence:
        return ContentEvidence(
            status="ok",
            score=0.7,
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
    """
    功能：构造与 detect orchestrator 一致的 trace_signature。

    Build trace signature aligned with detect orchestrator.

    Args:
        cfg: Configuration mapping.

    Returns:
        Trace signature mapping.
    """
    generation = cfg.get("generation") if isinstance(cfg.get("generation"), dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return {
        "num_inference_steps": cfg.get("inference_num_steps", generation.get("num_inference_steps", 16)),
        "guidance_scale": cfg.get("inference_guidance_scale", generation.get("guidance_scale", 7.0)),
        "height": cfg.get("inference_height", model.get("height", 512)),
        "width": cfg.get("inference_width", model.get("width", 512)),
    }


def _compute_planner_input_digest(
    planner: SubspacePlannerImpl,
    cfg: Dict[str, Any],
    trajectory_evidence: Dict[str, Any] | None,
    cfg_digest: str
) -> str:
    """
    功能：计算与运行期一致的 planner_input_digest。

    Compute planner_input_digest using planner runtime path.

    Args:
        planner: Planner implementation.
        cfg: Configuration mapping.
        trajectory_evidence: Optional trajectory evidence mapping.
        cfg_digest: Config digest string.

    Returns:
        Planner input digest string.
    """
    inputs: Dict[str, Any] = {"trace_signature": _build_trace_signature(cfg)}
    if trajectory_evidence is not None:
        inputs["trajectory_evidence"] = trajectory_evidence

    _ = cfg_digest
    return planner._build_planner_input_digest(inputs)


def test_detect_plan_mismatch_is_mismatch() -> None:
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

    cfg = {
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

    input_record = {
        "plan_digest": "embed_plan_digest_other",
        "basis_digest": "embed_basis_digest_other",
        "subspace_planner_impl_identity": {
            "impl_id": "subspace_planner_other_v1",
            "impl_version": "v1",
            "impl_digest": "other"
        }
    }

    record = run_detect_orchestrator(cfg, impl_set, input_record=input_record, cfg_digest="cfg_digest_test")

    assert record["plan_digest_validation_status"] == "mismatch"
    assert record["content_result"]["status"] == "mismatch"
    assert record["content_result"]["score"] is None
    assert record["plan_digest_mismatch_reason"] == "plan_digest_mismatch"
    assert record["content_result"].get("content_mismatch_field_path") == "content_evidence.plan_digest"

    fusion_result = record["fusion_result"]
    assert isinstance(fusion_result, FusionDecision)
    assert fusion_result.decision_status == "error"
    assert fusion_result.evidence_summary["content_score"] is None
    assert fusion_stub.called is False


def test_detect_trajectory_digest_mismatch_is_mismatch() -> None:
    """
    功能：轨迹摘要不一致必须触发 mismatch。

    Trajectory digest mismatch must force mismatch semantics.

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

    cfg = {
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

    detect_planner_input_digest = _compute_planner_input_digest(
        subspace_planner,
        cfg,
        {
            "status": "ok",
            "trajectory_spec_digest": "spec_digest_embed",
            "trajectory_digest": "traj_digest_detect"
        },
        "cfg_digest_test"
    )

    input_record = {
        "content_evidence_payload": {
            "trajectory_evidence": {
                "status": "ok",
                "trajectory_spec_digest": "spec_digest_embed",
                "trajectory_digest": "traj_digest_embed"
            }
        },
        "subspace_plan": {
            "verifiable_input_domain_spec": {
                "planner_input_digest": detect_planner_input_digest
            }
        }
    }

    trajectory_evidence = {
        "status": "ok",
        "trajectory_spec_digest": "spec_digest_embed",
        "trajectory_digest": "traj_digest_detect"
    }

    record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="cfg_digest_test",
        trajectory_evidence=trajectory_evidence
    )

    assert record["content_result"]["status"] == "mismatch"
    assert record["content_result"]["score"] is None
    assert record["content_result"].get("content_mismatch_reason") == "trajectory_digest_mismatch"
    assert record["content_result"].get("content_mismatch_field_path") == "content_evidence.trajectory_evidence.trajectory_digest"

    fusion_result = record["fusion_result"]
    assert isinstance(fusion_result, FusionDecision)
    assert fusion_result.decision_status == "error"
    assert fusion_stub.called is False


def test_detect_trajectory_absent_forces_absent() -> None:
    """
    功能：轨迹证据缺失必须触发 absent。

    Missing trajectory evidence must force absent semantics.

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

    cfg = {
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

    detect_planner_input_digest = _compute_planner_input_digest(
        subspace_planner,
        cfg,
        {
            "status": "ok",
            "trajectory_spec_digest": "spec_digest_detect",
            "trajectory_digest": "traj_digest_detect"
        },
        "cfg_digest_test"
    )

    input_record = {
        "content_evidence_payload": {
            "trajectory_evidence": {
                "status": "absent",
                "trajectory_spec_digest": None,
                "trajectory_digest": None
            }
        },
        "subspace_plan": {
            "verifiable_input_domain_spec": {
                "planner_input_digest": detect_planner_input_digest
            }
        }
    }

    trajectory_evidence = {
        "status": "ok",
        "trajectory_spec_digest": "spec_digest_detect",
        "trajectory_digest": "traj_digest_detect"
    }

    record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="cfg_digest_test",
        trajectory_evidence=trajectory_evidence
    )

    assert record["content_result"]["status"] == "absent"
    assert record["content_result"]["score"] is None

    fusion_result = record["fusion_result"]
    assert isinstance(fusion_result, FusionDecision)
    assert fusion_result.decision_status == "abstain"
    assert fusion_stub.called is False
