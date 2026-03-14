"""
File purpose: 覆盖 paper_full_cuda 在 sidecar 禁用条件下的 HF-only runtime 恢复路径回归测试。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.content_chain.unified_content_extractor import UnifiedContentExtractor
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision
from scripts import run_onefile_workflow


class _ContentExtractorSidecarDisabledStub:
    """功能：构造 sidecar 禁用时的 absent content payload。"""

    def extract(
        self,
        cfg: Dict[str, Any],
        cfg_digest: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = cfg_digest
        _ = inputs
        return {
            "status": "absent",
            "score": None,
            "score_parts": None,
            "lf_score": None,
            "hf_score": None,
            "content_failure_reason": "formal_profile_sidecar_disabled",
        }


class _FusionRuleStub:
    """功能：提供最小融合规则桩，避免测试落入无关逻辑。"""

    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
        _ = cfg
        return FusionDecision(
            is_watermarked=None,
            decision_status="abstain",
            thresholds_digest="t" * 64,
            evidence_summary={
                "content_score": content_evidence.get("score"),
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": "fusion_stub",
            },
            audit={"impl": "fusion_stub"},
        )


class _GeometryExtractorStub:
    """功能：返回 absent geometry，保持测试焦点在 content runtime。"""

    def extract(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "absent", "geo_score": None}


class _SyncStub:
    """功能：提供最小 sync 桩。"""

    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "ok"}


class _PlannerStub:
    """功能：返回同时包含 LF/HF basis 的最小 plan。"""

    impl_identity = {
        "impl_id": "subspace_planner",
        "impl_version": "v2",
        "impl_digest": "p" * 64,
    }

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: Optional[str] = None,
        cfg_digest: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        _ = cfg
        _ = mask_digest
        _ = cfg_digest
        _ = inputs
        return {
            "status": "ok",
            "plan": {
                "lf_basis": {"basis_id": "lf"},
                "hf_basis": {"basis_id": "hf"},
            },
            "plan_digest": "a" * 64,
            "basis_digest": "b" * 64,
            "audit": {},
        }


def test_prepare_detect_record_for_scoring_accepts_hf_score_when_sidecar_disabled(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda 在 sidecar 禁用且仅有 HF 分数时仍可生成 calibrate 输入。

    Verify that paper_full_cuda can recover a calibration input from an HF-only
    detect record when image-domain sidecar is intentionally disabled.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    detect_record = {
        "content_evidence_payload": {
            "status": "absent",
            "score": None,
            "score_parts": None,
            "detect_lf_score": None,
            "detect_hf_score": 0.91,
            "content_failure_reason": "formal_profile_sidecar_disabled",
        }
    }
    (records_dir / "detect_record.json").write_text(
        json.dumps(detect_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_path = run_onefile_workflow._prepare_detect_record_for_scoring(  # pyright: ignore[reportPrivateUsage]
        run_root,
        records_dir,
        run_onefile_workflow.PROFILE_PAPER_FULL_CUDA,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    content_payload = payload.get("content_evidence_payload")
    assert isinstance(content_payload, dict)
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == 0.91
    assert content_payload.get("calibration_sample_usage") == "formal_with_sidecar_disabled_marker"


def test_detect_runtime_mode_real_for_hf_only_when_sidecar_disabled(monkeypatch: Any) -> None:
    """
    功能：验证 sidecar 禁用且 HF trajectory 分数可用时 detect_runtime_mode 仍标记为 real。

    Verify that detect runtime remains real when sidecar is intentionally
    disabled but HF trajectory evidence is available.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    cache = LatentTrajectoryCache()
    cache.capture(0, [[[[1.0, 2.0, 3.0, 4.0]]]])

    monkeypatch.setattr(
        detect_orchestrator.detector_scoring,
        "extract_lf_score_from_detect_trajectory",
        lambda *args, **kwargs: (None, "lf_absent_for_sidecar_disabled"),
    )
    monkeypatch.setattr(
        detect_orchestrator.detector_scoring,
        "extract_hf_score_from_detect_trajectory",
        lambda *args, **kwargs: (0.97, "ok_trajectory_ok_exact"),
    )

    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "subspace": {"enabled": True},
            "hf": {"enabled": True},
            "lf": {"enabled": True},
        },
        "detect": {
            "content": {"enabled": True},
            "geometry": {"enabled": False},
        },
        "__detect_trajectory_latent_cache__": cache,
        "__pipeline_runtime_meta__": {"status": "built", "synthetic_pipeline": False},
    }

    input_record = {
        "plan_digest": "a" * 64,
        "basis_digest": "b" * 64,
        "subspace_planner_impl_identity": _PlannerStub.impl_identity,
    }
    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorSidecarDisabledStub(),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_PlannerStub(),
        sync_module=_SyncStub(),
    )

    record = detect_orchestrator.run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="c" * 64,
    )

    assert record.get("detect_runtime_mode") == "real"
    assert record.get("detect_runtime_is_fallback") is False
    content_payload = record.get("content_evidence_payload")
    assert isinstance(content_payload, dict)
    assert content_payload.get("detect_hf_score") == 0.97


def test_resolve_sidecar_disabled_reason_distinguishes_formal_and_ablation() -> None:
    """
    功能：验证 formal profile 与 ablation profile 的 sidecar 关闭原因被明确区分。

    Verify formal-profile and ablation sidecar-disabled reasons are distinct.

    Args:
        None.

    Returns:
        None.
    """
    assert detect_orchestrator._resolve_sidecar_disabled_reason(True, False) == "formal_profile_sidecar_disabled"
    assert detect_orchestrator._resolve_sidecar_disabled_reason(False, False) == "ablation_sidecar_disabled"


def test_detect_orchestrator_uses_lf_trajectory_evidence_when_sidecar_disabled(monkeypatch: Any) -> None:
    """
    功能：验证 sidecar 禁用时 LF trajectory 证据不会再被覆盖成 absent。

    Verify LF trajectory evidence remains canonical when image-domain sidecar is disabled.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        lambda **kwargs: (
            0.83,
            {
                "lf_status": "ok",
                "lf_trace_digest": "a" * 64,
                "bp_converged": True,
                "bp_iteration_count": 4,
                "parity_check_digest": "b" * 64,
                "lf_detect_path": "low_freq_template_trajectory",
            },
        ),
    )
    monkeypatch.setattr(
        detect_orchestrator,
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

    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "subspace": {"enabled": True},
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": True},
        },
        "detect": {
            "content": {"enabled": True},
            "geometry": {"enabled": False},
        },
        "__pipeline_runtime_meta__": {"status": "built", "synthetic_pipeline": False},
    }
    input_record = {
        "plan_digest": "a" * 64,
        "basis_digest": "b" * 64,
        "subspace_planner_impl_identity": _PlannerStub.impl_identity,
    }
    content_extractor = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="c" * 64,
    )
    impl_set = BuiltImplSet(
        content_extractor=content_extractor,
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_PlannerStub(),
        sync_module=_SyncStub(),
    )

    record = detect_orchestrator.run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="d" * 64,
    )

    content_payload = record.get("content_evidence_payload")
    assert isinstance(content_payload, dict)
    assert content_payload.get("status") == "ok"
    assert content_payload.get("lf_score") == 0.83
    lf_summary = content_payload.get("lf_evidence_summary")
    assert isinstance(lf_summary, dict)
    assert lf_summary.get("lf_status") == "ok"
    assert content_payload.get("content_failure_reason") is None


def test_detect_orchestrator_records_lf_trajectory_failure_reason_when_sidecar_disabled(monkeypatch: Any) -> None:
    """
    功能：验证 sidecar 禁用时 LF trajectory 失败原因会落盘到 detect record。

    Verify LF trajectory failure reasons remain auditable when sidecar is disabled.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        lambda **kwargs: (
            None,
            {
                "lf_status": "absent",
                "lf_absent_reason": "lf_projection_matrix_missing",
                "lf_detect_path": "low_freq_template_trajectory",
            },
        ),
    )
    monkeypatch.setattr(
        detect_orchestrator,
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

    cfg: Dict[str, Any] = {
        "paper_faithfulness": {"enabled": True},
        "watermark": {
            "subspace": {"enabled": True},
            "lf": {"enabled": True, "ecc": "sparse_ldpc"},
            "hf": {"enabled": True},
        },
        "detect": {
            "content": {"enabled": True},
            "geometry": {"enabled": False},
        },
        "__pipeline_runtime_meta__": {"status": "built", "synthetic_pipeline": False},
    }
    input_record = {
        "plan_digest": "a" * 64,
        "basis_digest": "b" * 64,
        "subspace_planner_impl_identity": _PlannerStub.impl_identity,
    }
    content_extractor = UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="c" * 64,
    )
    impl_set = BuiltImplSet(
        content_extractor=content_extractor,
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_PlannerStub(),
        sync_module=_SyncStub(),
    )

    record = detect_orchestrator.run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=input_record,
        cfg_digest="d" * 64,
    )

    content_payload = record.get("content_evidence_payload")
    assert isinstance(content_payload, dict)
    assert content_payload.get("status") == "absent"
    assert content_payload.get("content_failure_reason") == "lf_projection_matrix_missing"
    lf_summary = content_payload.get("lf_evidence_summary")
    assert isinstance(lf_summary, dict)
    assert lf_summary.get("lf_absent_reason") == "lf_projection_matrix_missing"