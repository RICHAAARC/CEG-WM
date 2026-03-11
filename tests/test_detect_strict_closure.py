"""
File purpose: Detect strict closure regression tests.
Module type: General module
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from main.cli.run_detect import assert_detect_runtime_dependencies
from main.core import digests
from main.registries.runtime_resolver import BuiltImplSet
from main.watermarking.detect.orchestrator import run_detect_orchestrator, _bind_scores_if_ok, _resolve_expected_plan_digest
from main.watermarking.fusion.interfaces import FusionDecision


class _ContentExtractorStub:
    def __init__(self, status: str = "ok", score: float | None = 0.7) -> None:
        self._status = status
        self._score = score

    def extract(self, cfg: Dict[str, Any], inputs: Dict[str, Any] | None = None, cfg_digest: str | None = None) -> Dict[str, Any]:
        _ = cfg
        _ = inputs
        _ = cfg_digest
        return {
            "status": self._status,
            "score": self._score if self._status == "ok" else None,
            "plan_digest": "plan_detect",
            "basis_digest": "basis_detect",
            "content_failure_reason": None if self._status == "ok" else "detector_extraction_failed",
            "score_parts": {"lf_score": self._score, "hf_score": None} if self._status == "ok" else None,
            "lf_score": self._score if self._status == "ok" else None,
            "hf_score": None,
            "audit": {
                "impl_identity": "content_stub",
                "impl_version": "v1",
                "impl_digest": digests.canonical_sha256({"name": "content_stub"}),
                "trace_digest": digests.canonical_sha256({"trace": "content_stub"}),
            },
        }


class _GeometryExtractorStub:
    def extract(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "absent", "geo_score": None}


class _FusionRuleStub:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
        _ = cfg
        return FusionDecision(
            is_watermarked=False,
            decision_status="decided",
            thresholds_digest="stub",
            evidence_summary={
                "content_score": content_evidence.get("score"),
                "geometry_score": geometry_evidence.get("geo_score"),
                "content_status": content_evidence.get("status", "absent"),
                "geometry_status": geometry_evidence.get("status", "absent"),
                "fusion_rule_id": "fusion_stub",
            },
            audit={"impl": "fusion_stub"},
        )


class _SyncStub:
    def sync(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        _ = cfg
        return {"status": "ok"}


class _SubspacePlannerStub:
    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: str | None = None,
        cfg_digest: str | None = None,
        inputs: Dict[str, Any] | None = None,
    ) -> Any:
        _ = cfg
        _ = mask_digest
        _ = cfg_digest
        _ = inputs
        payload = {
            "plan": {
                "planner_input_digest": "planner_input_digest_ok",
                "planner_impl_identity": {
                    "impl_id": "subspace_planner_v2",
                    "impl_version": "v2",
                    "impl_digest": "digest",
                },
            },
            "plan_digest": "plan_detect",
            "basis_digest": "basis_detect",
        }
        return SimpleNamespace(
            plan=payload["plan"],
            plan_digest=payload["plan_digest"],
            basis_digest=payload["basis_digest"],
            as_dict=lambda: payload,
        )


def _build_impl_set(content_status: str = "ok", content_score: float | None = 0.7) -> BuiltImplSet:
    return BuiltImplSet(
        content_extractor=_ContentExtractorStub(status=content_status, score=content_score),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=_SubspacePlannerStub(),
        sync_module=_SyncStub(),
    )


def _build_cfg(**overrides: Any) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "evaluate": {"target_fpr": 1e-6},
        "watermark": {
            "plan_digest": "plan_cfg",
            "hf": {"enabled": False},
            "subspace": {"enabled": True},
        },
        "detect": {"content": {"enabled": True}},
    }
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _build_matching_input_record() -> Dict[str, Any]:
    return {
        "plan_digest": "plan_detect",
        "basis_digest": "basis_detect",
        "subspace_planner_impl_identity": {
            "impl_id": "subspace_planner_v2",
            "impl_version": "v2",
            "impl_digest": "digest",
        },
        "content_evidence_payload": {
            "trajectory_evidence": {
                "status": "ok",
                "trajectory_spec_digest": "spec_digest_ok",
                "trajectory_digest": "traj_digest_ok",
            }
        },
        "subspace_plan": {
            "verifiable_input_domain_spec": {
                "planner_input_digest": "planner_input_digest_ok",
            }
        },
    }


def _build_matching_trajectory_evidence() -> Dict[str, Any]:
    return {
        "status": "ok",
        "trajectory_spec_digest": "spec_digest_ok",
        "trajectory_digest": "traj_digest_ok",
    }


def test_detect_missing_pipeline_is_fail_fast_by_default() -> None:
    cfg = _build_cfg()
    with pytest.raises(RuntimeError, match="detect_missing_pipeline_dependency"):
        assert_detect_runtime_dependencies(cfg, {"test_mode": False}, pipeline_obj=None)


def test_allow_missing_pipeline_only_in_test_mode_or_explicit_flag() -> None:
    cfg_test_mode = _build_cfg(detect={"content": {"enabled": True}, "runtime": {"test_mode": True}})
    decision_test_mode = assert_detect_runtime_dependencies(cfg_test_mode, {"test_mode": False}, pipeline_obj=None)
    assert decision_test_mode["allow_missing_pipeline_for_detect"] is True

    cfg_allow_flag = _build_cfg(runtime={"allow_missing_pipeline_for_detect": True})
    decision_allow_flag = assert_detect_runtime_dependencies(cfg_allow_flag, {"test_mode": False}, pipeline_obj=None)
    assert decision_allow_flag["allow_missing_pipeline_for_detect"] is True

    record = run_detect_orchestrator(
        _build_cfg(),
        _build_impl_set(),
        input_record={"plan_digest": "plan_detect"},
        cfg_digest="cfg_digest",
        trajectory_evidence={"status": "absent", "trajectory_absent_reason": "inference_failed"},
    )
    payload = record["content_evidence_payload"]
    assert payload["status"] == "absent"
    assert payload.get("score") is None
    assert payload.get("lf_score") is None
    assert payload.get("hf_score") is None


def test_missing_expected_plan_digest_is_absent_and_no_scores() -> None:
    cfg = _build_cfg()
    record = run_detect_orchestrator(cfg, _build_impl_set(), input_record={}, cfg_digest="cfg_digest")
    payload = record["content_evidence_payload"]
    assert payload["status"] == "absent"
    assert payload.get("content_failure_reason") == "detector_no_plan_expected"
    assert payload.get("score") is None
    assert payload.get("lf_score") is None
    assert payload.get("hf_score") is None

    cfg_test_mode = _build_cfg(detect={"content": {"enabled": True}, "runtime": {"test_mode": True}})
    record_test_mode = run_detect_orchestrator(cfg_test_mode, _build_impl_set(), input_record={}, cfg_digest="cfg_digest")
    assert record_test_mode["allow_cfg_plan_digest_fallback_used"] is True


def test_resolve_expected_plan_digest_from_embed_trace_injection_evidence() -> None:
    expected_plan_digest = "plan_digest_from_embed_trace_injection"
    input_record = {
        "embed_trace": {
            "injection_evidence": {
                "status": "ok",
                "plan_digest": expected_plan_digest,
            }
        }
    }

    resolved = _resolve_expected_plan_digest(input_record)
    assert resolved == expected_plan_digest


def test_plan_digest_mismatch_short_circuits_scoring() -> None:
    cfg = _build_cfg()
    input_record = _build_matching_input_record()
    input_record["plan_digest"] = "plan_embed_other"

    record = run_detect_orchestrator(
        cfg,
        _build_impl_set(),
        input_record=input_record,
        cfg_digest="cfg_digest",
        trajectory_evidence=_build_matching_trajectory_evidence(),
    )
    payload = record["content_evidence_payload"]
    assert payload["status"] == "mismatch"
    assert payload.get("score") is None
    assert payload.get("lf_score") is None
    assert payload.get("hf_score") is None
    assert payload.get("score_parts") is None


def test_scores_written_only_when_status_ok() -> None:
    for failing_status in ["absent", "mismatch", "failed"]:
        payload = {
            "status": failing_status,
            "score": 0.5,
            "lf_score": 0.4,
            "hf_score": 0.3,
            "score_parts": {
                "lf_score": 0.4,
                "hf_score": 0.3,
                "content_score": 0.5,
            },
        }
        _bind_scores_if_ok(payload)
        assert payload["score"] is None
        assert payload["lf_score"] is None
        assert payload["hf_score"] is None
        if isinstance(payload.get("score_parts"), dict):
            assert payload["score_parts"]["lf_score"] is None
            assert payload["score_parts"]["hf_score"] is None
            assert payload["score_parts"]["content_score"] is None

    ok_payload = {
        "status": "ok",
        "score": 0.6,
        "lf_score": 0.6,
        "hf_score": None,
        "score_parts": {
            "lf_score": 0.6,
            "hf_score": "<absent>",
        },
    }
    _bind_scores_if_ok(ok_payload)
    assert ok_payload["score"] == 0.6
    assert ok_payload["lf_score"] == 0.6
    assert ok_payload["status"] == "ok"

