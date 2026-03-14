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
from main.policy.runtime_whitelist import load_policy_path_semantics
from main.registries.fusion_registry import resolve_fusion_rule
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
            "content_score": self._score if self._status == "ok" else None,
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


class _PositiveFusionRuleStub:
    def fuse(self, cfg: Dict[str, Any], content_evidence: Dict[str, Any], geometry_evidence: Dict[str, Any]) -> FusionDecision:
        _ = cfg
        return FusionDecision(
            is_watermarked=True,
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
                    "impl_id": "subspace_planner",
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


def _build_formal_cfg() -> Dict[str, Any]:
    return _build_cfg(
        policy_path="content_np_geo_rescue",
        paper_faithfulness={"enabled": True},
        detect={
            "content": {"enabled": True},
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "sync_primary_anchor_secondary": True,
            },
        },
        __detect_pipeline_obj__=object(),
        __runtime_self_attention_maps__=[0],
    )


def _build_matching_input_record() -> Dict[str, Any]:
    return {
        "plan_digest": "plan_detect",
        "basis_digest": "basis_detect",
        "subspace_planner_impl_identity": {
            "impl_id": "subspace_planner",
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
        assert_detect_runtime_dependencies(cfg, {}, pipeline_obj=None)


def test_allow_missing_pipeline_only_with_explicit_flag() -> None:
    # test_mode 不再放行缺失 pipeline，只有显式 allow_flag 可通过校验。
    cfg_test_mode = _build_cfg(detect={"content": {"enabled": True}, "runtime": {"test_mode": True}})
    with pytest.raises(RuntimeError, match="detect_missing_pipeline_dependency"):
        assert_detect_runtime_dependencies(cfg_test_mode, {}, pipeline_obj=None)

    cfg_allow_flag = _build_cfg(runtime={"allow_missing_pipeline_for_detect": True})
    decision_allow_flag = assert_detect_runtime_dependencies(cfg_allow_flag, {}, pipeline_obj=None)
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
    # formal path 语义闭包清理：不再从 cfg 回填 expected_plan_digest，test_mode 路径与正式路径统一


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


@pytest.mark.parametrize(
    ("geometry_status", "failure_field", "failure_reason"),
    [
        ("absent", "geometry_absent_reason", "detect_sync_relation_binding_missing"),
        ("mismatch", "geometry_failure_reason", "detect_sync_mismatch"),
    ],
)
def test_formal_optional_geometry_chain_keeps_content_positive_decision(
    monkeypatch: pytest.MonkeyPatch,
    geometry_status: str,
    failure_field: str,
    failure_reason: str,
) -> None:
    cfg = _build_formal_cfg()
    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(status="ok", score=0.7),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_PositiveFusionRuleStub(),
        subspace_planner=_SubspacePlannerStub(),
        sync_module=_SyncStub(),
    )

    def _stub_geometry_chain(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "status": geometry_status,
            "geo_score": None,
            failure_field: failure_reason,
        }

    monkeypatch.setattr("main.watermarking.detect.orchestrator._run_geometry_chain_with_sync", _stub_geometry_chain)

    record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=_build_matching_input_record(),
        cfg_digest="cfg_digest",
        trajectory_evidence=_build_matching_trajectory_evidence(),
    )

    fusion_result = record["fusion_result"]
    assert isinstance(fusion_result, FusionDecision)
    assert fusion_result.decision_status == "decided"
    assert fusion_result.is_watermarked is True
    assert fusion_result.audit.get("failure_reason") is None
    assert record["execution_report"]["geometry_chain_status"] == ("absent" if geometry_status == "absent" else "failed")
    assert record["final_decision"]["is_watermarked"] is True


def test_formal_optional_geometry_failed_keeps_real_fusion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build_formal_cfg()
    factory = resolve_fusion_rule("fusion_neyman_pearson")
    fusion_rule = factory({})
    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(status="ok", score=0.7),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=fusion_rule,
        subspace_planner=_SubspacePlannerStub(),
        sync_module=_SyncStub(),
    )

    monkeypatch.setattr(
        "main.watermarking.detect.orchestrator._run_geometry_chain_with_sync",
        lambda *args, **kwargs: {"status": "failed", "geo_score": None, "geometry_failure_reason": "detect_sync_failed"},
    )

    record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=_build_matching_input_record(),
        cfg_digest="cfg_digest",
        trajectory_evidence=_build_matching_trajectory_evidence(),
    )

    fusion_result = record["fusion_result"]
    assert isinstance(fusion_result, FusionDecision)
    assert fusion_result.decision_status == "error"
    assert fusion_result.is_watermarked is None
    assert fusion_result.audit.get("failure_reason") == "geometry_fail"
    assert record["final_decision"]["is_watermarked"] is None


def test_non_formal_path_keeps_internal_content_only_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build_cfg(
        policy_path="content_only",
        detect={
            "content": {"enabled": True},
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
            },
        },
        __detect_pipeline_obj__=object(),
    )
    impl_set = BuiltImplSet(
        content_extractor=_ContentExtractorStub(status="ok", score=0.7),
        geometry_extractor=_GeometryExtractorStub(),
        fusion_rule=_PositiveFusionRuleStub(),
        subspace_planner=_SubspacePlannerStub(),
        sync_module=_SyncStub(),
    )

    monkeypatch.setattr(
        "main.watermarking.detect.orchestrator._run_geometry_chain_with_sync",
        lambda *args, **kwargs: {"status": "absent", "geo_score": None, "geometry_absent_reason": "test_absent"},
    )

    record = run_detect_orchestrator(
        cfg,
        impl_set,
        input_record=_build_matching_input_record(),
        cfg_digest="cfg_digest",
        trajectory_evidence=_build_matching_trajectory_evidence(),
    )

    fusion_result = record["fusion_result"]
    assert isinstance(fusion_result, FusionDecision)
    assert fusion_result.decision_status == "decided"
    assert fusion_result.is_watermarked is True


def test_formal_required_chain_closure_follows_loaded_policy_semantics(
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
        semantics_path = tmp_path / "policy_path_semantics.yaml"
        semantics_path.write_text(
                """
policy_path_semantics_version: "test"
policy_paths:
    alternate_geometry_required:
        decoder_type: "content_correlation"
        description: "Alternate geometry-required path"
        required_chains:
            content: true
            geometry: true
        optional_chains:
            geometry: false
        on_chain_failure:
            content:
                action: "failed"
                set_decision_to: null
                record_fail_reason: "content_chain_required_but_unavailable"
            geometry:
                action: "failed"
                set_decision_to: null
                record_fail_reason: "alternate_geometry_chain_required"
        audit_obligations:
            record_fields_level: "core_required"
            path_required_fields: {}
            recommended_fields: []
""".strip(),
                encoding="utf-8",
        )
        semantics = load_policy_path_semantics(str(semantics_path), allow_non_authoritative=True)
        cfg = _build_cfg(
                policy_path="alternate_geometry_required",
                paper_faithfulness={"enabled": True},
                detect={
                        "content": {"enabled": True},
                        "geometry": {
                                "enabled": True,
                                "enable_attention_anchor": True,
                                "sync_primary_anchor_secondary": True,
                        },
                },
                __detect_pipeline_obj__=object(),
                __runtime_self_attention_maps__=[0],
                __policy_path_semantics__=semantics,
        )
        impl_set = BuiltImplSet(
                content_extractor=_ContentExtractorStub(status="ok", score=0.7),
                geometry_extractor=_GeometryExtractorStub(),
                fusion_rule=_PositiveFusionRuleStub(),
                subspace_planner=_SubspacePlannerStub(),
                sync_module=_SyncStub(),
        )

        monkeypatch.setattr(
                "main.watermarking.detect.orchestrator._run_geometry_chain_with_sync",
                lambda *args, **kwargs: {"status": "absent", "geo_score": None, "geometry_absent_reason": "test_absent"},
        )

        record = run_detect_orchestrator(
                cfg,
                impl_set,
                input_record=_build_matching_input_record(),
                cfg_digest="cfg_digest",
                trajectory_evidence=_build_matching_trajectory_evidence(),
        )

        fusion_result = record["fusion_result"]
        assert isinstance(fusion_result, FusionDecision)
        assert fusion_result.decision_status == "error"
        assert fusion_result.is_watermarked is None
        assert fusion_result.audit.get("failure_reason") == "alternate_geometry_chain_required"
        assert fusion_result.audit.get("formal_policy_path") == "alternate_geometry_required"
        assert fusion_result.audit.get("formal_required_chain") == "geometry"
        assert record["final_decision"]["is_watermarked"] is None


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

