"""
File purpose: HF robust channel regression tests.
Module type: General module

功能说明：
- 覆盖 HF disabled / mismatch / failed 语义。
- 验证 HF 参数进入 plan_digest 输入域并触发摘要变化。
- 验证新增字段未注册时审计失败。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from scripts.audits.audit_records_fields_append_only import run_audit
from main.core import digests
from main.watermarking.content_chain.interfaces import ContentEvidence
from main.watermarking.content_chain.unified_content_extractor import UnifiedContentExtractor
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
)


def _build_detector() -> UnifiedContentExtractor:
    return UnifiedContentExtractor(
        impl_id="unified_content_extractor",
        impl_version="v2",
        impl_digest="test_impl_digest_s05",
    )


def _build_cfg(hf_enabled: bool = False) -> Dict[str, Any]:
    return {
        "detect": {
            "content": {
                "enabled": True,
            }
        },
        "watermark": {
            "plan_digest": "plan_digest_s05",
            "hf": {
                "enabled": hf_enabled,
            },
        },
    }


def _build_lf_ok(score: float = 0.81) -> ContentEvidence:
    return ContentEvidence(
        status="ok",
        score=score,
        audit={
            "impl_identity": "low_freq_template_codec",
            "impl_version": "v1",
            "impl_digest": "digest",
            "trace_digest": "trace",
        },
        lf_score=score,
    )


def test_hf_disabled_absent_keeps_lf_score_unchanged() -> None:
    cfg = _build_cfg(hf_enabled=False)
    detector = _build_detector()

    lf_evidence = _build_lf_ok(score=0.66)
    hf_evidence: Dict[str, Any] = {
        "status": "absent",
        "hf_score": None,
        "hf_evidence_summary": {
            "hf_status": "absent",
            "hf_absent_reason": "hf_disabled_by_config",
        },
    }

    result = detector.extract(
        cfg=cfg,
        inputs={
            "plan_digest": "plan_digest_s05",
            "lf_evidence": lf_evidence,
            "hf_evidence": hf_evidence,
        },
    )

    assert result.status == "ok"
    assert result.hf_score is None
    assert result.lf_score == 0.66
    assert result.score == 0.66
    assert isinstance(result.score_parts, dict)
    assert result.score_parts.get("hf_status") == "absent"
    assert result.score_parts.get("hf_absent_reason") == "hf_disabled_by_config"


def test_hf_parameter_drift_changes_plan_digest_binding() -> None:
    impl_digest = digests.canonical_sha256({"impl_id": SUBSPACE_PLANNER_ID, "impl_version": SUBSPACE_PLANNER_VERSION})
    planner = SubspacePlannerImpl(SUBSPACE_PLANNER_ID, SUBSPACE_PLANNER_VERSION, impl_digest)

    cfg_base: Dict[str, Any] = {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 8,
                "feature_dim": 16,
                "seed": 9,
                "timestep_start": 0,
                "timestep_end": 6,
            },
            "hf": {
                "enabled": True,
                "codebook_id": "hf_codebook_v1",
                "ecc": 2,
                "tail_truncation_ratio": 0.10,
                "tail_truncation_mode": "gaussian",
            },
        }
    }
    cfg_changed: Dict[str, Any] = {
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 8,
                "feature_dim": 16,
                "seed": 9,
                "timestep_start": 0,
                "timestep_end": 6,
            },
            "hf": {
                "enabled": True,
                "codebook_id": "hf_codebook_v1",
                "ecc": 2,
                "tail_truncation_ratio": 0.35,
                "tail_truncation_mode": "gaussian",
            },
        }
    }

    inputs: Dict[str, Any] = {
        "trace_signature": {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "height": 512,
            "width": 512,
        }
    }

    result_base = planner.plan(cfg_base, mask_digest="mask_digest_s05", cfg_digest="cfg_digest_s05", inputs=inputs)
    result_changed = planner.plan(cfg_changed, mask_digest="mask_digest_s05", cfg_digest="cfg_digest_s05", inputs=inputs)

    assert result_base.status == "ok"
    assert result_changed.status == "ok"
    assert result_base.plan_digest != result_changed.plan_digest


def test_plan_digest_mismatch_blocks_content_score() -> None:
    cfg = _build_cfg(hf_enabled=True)
    detector = _build_detector()

    lf_evidence = _build_lf_ok(score=0.58)
    hf_evidence: Dict[str, Any] = {
        "status": "mismatch",
        "hf_score": None,
        "hf_evidence_summary": {
            "hf_status": "mismatch",
            "hf_failure_reason": "hf_plan_mismatch",
        },
        "content_failure_reason": "hf_plan_mismatch",
    }

    result = detector.extract(
        cfg=cfg,
        inputs={
            "plan_digest": "plan_digest_s05",
            "lf_evidence": lf_evidence,
            "hf_evidence": hf_evidence,
        },
    )

    assert result.status == "mismatch"
    assert result.score is None
    assert result.content_failure_reason == "hf_plan_mismatch"


def test_hf_failure_does_not_mutate_lf_evidence() -> None:
    cfg = _build_cfg(hf_enabled=True)
    detector = _build_detector()

    lf_evidence = _build_lf_ok(score=0.73)
    hf_evidence: Dict[str, Any] = {
        "status": "failed",
        "hf_score": None,
        "hf_evidence_summary": {
            "hf_status": "failed",
            "hf_failure_reason": "hf_detection_failed",
        },
        "content_failure_reason": "hf_detection_failed",
    }

    result = detector.extract(
        cfg=cfg,
        inputs={
            "plan_digest": "plan_digest_s05",
            "lf_evidence": lf_evidence,
            "hf_evidence": hf_evidence,
        },
    )

    assert result.status == "failed"
    assert result.score is None
    assert result.content_failure_reason == "hf_detection_failed"
    assert lf_evidence.lf_score == 0.73


def test_new_hf_field_unregistered_must_fail_audit(tmp_path: Path) -> None:
    repo_root = tmp_path
    configs_dir = repo_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    extensions_yaml = """
version: "v1"
append_only: true
fields:
  - path: "content_evidence.hf_evidence_summary"
    layer: "diagnostic"
    type: "metrics_dict"
    required: false
    missing_semantics: "absent_ok"
    description: "hf summary"
""".strip()

    contracts_yaml = """
records_schema:
  field_paths_registry:
    - schema_version
    - content_evidence
""".strip()

    (configs_dir / "records_schema_extensions.yaml").write_text(extensions_yaml, encoding="utf-8")
    (configs_dir / "frozen_contracts.yaml").write_text(contracts_yaml, encoding="utf-8")

    result = run_audit(repo_root)
    assert result["result"] == "FAIL"
    assert "content_evidence.hf_evidence_summary" in result["evidence"]["missing_in_registry"]
