"""
File purpose: Validate attacked-positive main dispatch routing in detect orchestrator.
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
from PIL import Image
import pytest

from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.registries.runtime_resolver import BuiltImplSet
import main.watermarking.detect.orchestrator as detect_orchestrator
from main.watermarking.fusion.interfaces import FusionDecision


class _CapturingContentExtractor:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        cfg_digest: str | None = None,
    ) -> Dict[str, Any]:
        detector_inputs = dict(inputs) if isinstance(inputs, dict) else {}
        self.calls.append(
            {
                "cfg": dict(cfg),
                "inputs": detector_inputs,
                "cfg_digest": cfg_digest,
            }
        )
        lf_trace_node = detector_inputs.get("lf_detect_trace")
        lf_trace = cast(Dict[str, Any], lf_trace_node) if isinstance(lf_trace_node, dict) else {}
        lf_evidence_node = detector_inputs.get("lf_evidence")
        lf_evidence = cast(Dict[str, Any], lf_evidence_node) if isinstance(lf_evidence_node, dict) else {}
        lf_status = lf_trace.get("lf_status") if isinstance(lf_trace.get("lf_status"), str) else "failed"

        if lf_status == "ok":
            lf_score_node = lf_evidence.get("lf_score")
            lf_score = float(lf_score_node) if isinstance(lf_score_node, (int, float)) else 0.0
            return {
                "status": "ok",
                "score": lf_score,
                "content_chain_score": lf_score,
                "lf_score": lf_score,
                "hf_score": None,
                "score_parts": {},
                "content_failure_reason": None,
                "audit": {
                    "impl_identity": "capturing_content_extractor",
                    "impl_version": "v1",
                    "impl_digest": "capturing-content-extractor-digest",
                    "trace_digest": "capturing-content-extractor-trace",
                },
            }

        failure_reason = None
        if isinstance(lf_trace.get("lf_failure_reason"), str):
            failure_reason = lf_trace.get("lf_failure_reason")
        elif isinstance(lf_trace.get("lf_absent_reason"), str):
            failure_reason = lf_trace.get("lf_absent_reason")

        return {
            "status": lf_status,
            "score": None,
            "content_chain_score": None,
            "lf_score": None,
            "hf_score": None,
            "score_parts": {},
            "content_failure_reason": failure_reason,
            "audit": {
                "impl_identity": "capturing_content_extractor",
                "impl_version": "v1",
                "impl_digest": "capturing-content-extractor-digest",
                "trace_digest": "capturing-content-extractor-trace",
            },
        }


class _UnusedGeometryExtractor:
    def extract(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("geometry extractor should be bypassed by ablation gating")


class _UnusedSyncModule:
    def sync(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("sync module should be bypassed by ablation gating")


class _FusionRuleStub:
    def fuse(
        self,
        cfg: Dict[str, Any],
        content_evidence: Dict[str, Any],
        geometry_evidence: Dict[str, Any],
    ) -> FusionDecision:
        _ = cfg
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


def _build_detect_cfg() -> Dict[str, Any]:
    """
    功能：构造 attacked_positive 主分发测试的最小 detect 配置。

    Build the minimal detect configuration for attacked-positive dispatch tests.

    Args:
        None.

    Returns:
        Detect configuration mapping.
    """
    return {
        "paper_faithfulness": {
            "enabled": True,
        },
        "detect_runtime": {
            "image_domain_sidecar_enabled": False,
        },
        "ablation": {
            "normalized": {
                "enable_geometry": False,
            }
        },
        "attestation": {
            "enabled": False,
        },
        "watermark": {
            "key_id": "k-test",
            "pattern_id": "p-test",
            "lf": {
                "enabled": True,
                "ecc": "sparse_ldpc",
            },
            "hf": {
                "enabled": False,
            },
        },
    }


def _build_plan_override() -> Dict[str, Any]:
    """
    功能：构造主分发测试所需的最小 detect plan override。

    Build the minimal detect-plan override required by the dispatch tests.

    Args:
        None.

    Returns:
        Detect-plan override mapping.
    """
    planner_identity = {
        "impl_id": "planner_impl",
        "impl_version": "v1",
        "impl_digest": "planner-digest",
    }
    return {
        "status": "ok",
        "plan_digest": "plan-digest",
        "basis_digest": "basis-digest",
        "plan_failure_reason": None,
        "plan": {
            "planner_impl_identity": dict(planner_identity),
            "lf_basis": {
                "trajectory_feature_spec": {
                    "feature_operator": "masked_normalized_random_projection",
                    "edit_timestep": 0,
                },
            },
            "band_spec": {},
        },
    }


def _build_input_record(image_path: Path, sample_role: str) -> Dict[str, Any]:
    """
    功能：构造 attacked_positive 主分发测试输入记录。

    Build the detect input record fixture for attacked-positive dispatch tests.

    Args:
        image_path: Detect input image path.
        sample_role: Detect sample role.

    Returns:
        Input record mapping.
    """
    planner_identity = {
        "impl_id": "planner_impl",
        "impl_version": "v1",
        "impl_digest": "planner-digest",
    }
    return {
        "operation": "embed",
        "sample_role": sample_role,
        "watermarked_path": image_path.as_posix(),
        "image_path": image_path.as_posix(),
        "inputs": {
            "input_image_path": image_path.as_posix(),
        },
        "plan_digest": "plan-digest",
        "basis_digest": "basis-digest",
        "subspace_planner_impl_identity": dict(planner_identity),
        "subspace_plan": {
            "planner_impl_identity": dict(planner_identity),
        },
    }


def _build_latent_cache(fill_value: float) -> LatentTrajectoryCache:
    """
    功能：构造单 timestep latent cache。

    Build a single-timestep latent cache fixture.

    Args:
        fill_value: Fill value used in the cached tensor.

    Returns:
        Latent trajectory cache.
    """
    cache = LatentTrajectoryCache()
    cache.capture(0, np.full((1, 1, 2, 4), fill_value, dtype=np.float32))
    return cache


def _write_test_image(tmp_path: Path, file_name: str) -> Path:
    """
    功能：写入主分发测试使用的最小图像夹具。

    Write a minimal image fixture for the dispatch tests.

    Args:
        tmp_path: Pytest temporary directory.
        file_name: Output file name.

    Returns:
        Written image path.
    """
    image_path = tmp_path / file_name
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(image_path)
    return image_path


def _build_impl_set(content_extractor: _CapturingContentExtractor) -> BuiltImplSet:
    """
    功能：构造主分发测试的最小实现集合。

    Build the minimal implementation set for the dispatch tests.

    Args:
        content_extractor: Capturing content extractor stub.

    Returns:
        Built implementation set.
    """
    return BuiltImplSet(
        content_extractor=content_extractor,
        geometry_extractor=_UnusedGeometryExtractor(),
        fusion_rule=_FusionRuleStub(),
        subspace_planner=object(),
        sync_module=_UnusedSyncModule(),
    )


def _extract_record_lf_trace(record: Dict[str, Any]) -> Dict[str, Any]:
    content_payload = cast(Dict[str, Any], record["content_evidence_payload"])
    score_parts = cast(Dict[str, Any], content_payload["score_parts"])
    return cast(Dict[str, Any], score_parts["lf_trajectory_detect_trace"])


def _extract_dispatch_call(content_extractor: _CapturingContentExtractor) -> Dict[str, Any]:
    dispatch_calls = [
        call
        for call in content_extractor.calls
        if isinstance(call.get("inputs"), dict)
        and "lf_detect_trace" in cast(Dict[str, Any], call["inputs"])
    ]
    assert len(dispatch_calls) == 1
    return cast(Dict[str, Any], dispatch_calls[0])


def test_run_detect_orchestrator_attacked_positive_uses_image_conditioned_main_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 paper 正式模式下 attacked_positive 主分发优先走 image-conditioned LF。

    Ensure paper-mode attacked-positive samples enter the dedicated
    image-conditioned LF main-dispatch branch instead of the default trajectory
    branch.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    attacked_image_path = _write_test_image(tmp_path, "attacked_main_dispatch.png")
    attacked_cache = _build_latent_cache(3.0)
    content_extractor = _CapturingContentExtractor()
    captures: List[Dict[str, Any]] = []

    def fake_extract_lf_raw_score_from_trajectory(**kwargs: Any) -> tuple[float, Dict[str, Any]]:
        captures.append(dict(kwargs))
        return 0.73, {
            "lf_status": "ok",
            "lf_detect_path": str(kwargs.get("detect_path", "low_freq_template_trajectory")),
            "detect_variant": "correlation_v2",
            "message_source": "attestation_event_digest",
            "n_bits_compared": 96,
            "agreement_count": 84,
            "codeword_agreement": 0.875,
            "bit_error_count": 12,
            "decoded_bits": [1 if index % 2 == 0 else -1 for index in range(96)],
            "mismatch_indices": [1, 5, 9],
        }

    def fail_image_sidecar(*args: Any, **kwargs: Any) -> tuple[float | None, float | None, Dict[str, Any]]:
        raise AssertionError("attacked_positive main dispatch must not route through image sidecar helper")

    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        fake_extract_lf_raw_score_from_trajectory,
    )
    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_content_raw_scores_from_image",
        fail_image_sidecar,
    )

    cfg = _build_detect_cfg()
    cfg["__lf_attacked_image_conditioned_latent_cache__"] = attacked_cache
    cfg["__lf_formal_exact_context__"] = {
        "formal_exact_evidence_source": "input_image_conditioned_reconstruction",
        "formal_exact_object_binding_status": "ok",
        "formal_exact_image_path_source": "input_record.watermarked_path",
        "image_conditioned_reconstruction_available": True,
        "image_conditioned_reconstruction_status": "ok",
    }

    record = detect_orchestrator.run_detect_orchestrator(
        cfg=cfg,
        impl_set=_build_impl_set(content_extractor),
        input_record=_build_input_record(attacked_image_path, "attacked_positive"),
        cfg_digest="cfg-digest",
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=None,
        detect_plan_result_override=_build_plan_override(),
    )

    assert len(captures) == 1
    assert captures[0]["latent_cache"] is attacked_cache
    assert captures[0]["detect_path"] == "low_freq_template_image_conditioned_attack"
    dispatch_call = _extract_dispatch_call(content_extractor)
    detector_inputs = cast(Dict[str, Any], dispatch_call["inputs"])
    lf_trace = cast(Dict[str, Any], detector_inputs["lf_detect_trace"])
    assert lf_trace["lf_detect_path"] == "low_freq_template_image_conditioned_attack"
    assert lf_trace["formal_exact_object_binding_status"] == "ok"
    assert lf_trace["image_conditioned_reconstruction_status"] == "ok"
    record_lf_trace = _extract_record_lf_trace(record)
    assert record_lf_trace["lf_detect_path"] == "low_freq_template_image_conditioned_attack"
    assert record_lf_trace["n_bits_compared"] == 96
    assert record_lf_trace["agreement_count"] == 84
    assert record_lf_trace["codeword_agreement"] == pytest.approx(0.875)
    content_payload = cast(Dict[str, Any], record["content_evidence_payload"])
    lf_summary = cast(Dict[str, Any], content_payload["lf_evidence_summary"])
    assert lf_summary["detect_variant"] == "correlation_v2"
    assert lf_summary["message_source"] == "attestation_event_digest"
    assert lf_summary["n_bits_compared"] == 96
    assert lf_summary["agreement_count"] == 84
    assert lf_summary["codeword_agreement"] == pytest.approx(0.875)


def test_run_detect_orchestrator_attacked_positive_fail_closes_without_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 attacked_positive 主分发在缺失 image-conditioned cache 时 fail-closed。

    Ensure attacked-positive main dispatch fails closed when the
    image-conditioned latent cache is missing and never falls back to the
    default trajectory branch.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    attacked_image_path = _write_test_image(tmp_path, "attacked_missing_cache_main_dispatch.png")
    content_extractor = _CapturingContentExtractor()

    def fail_trajectory(*args: Any, **kwargs: Any) -> tuple[float, Dict[str, Any]]:
        raise AssertionError("attacked_positive main dispatch must not fall back to default trajectory scoring")

    def fail_image_sidecar(*args: Any, **kwargs: Any) -> tuple[float | None, float | None, Dict[str, Any]]:
        raise AssertionError("attacked_positive main dispatch must not route through image sidecar helper")

    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        fail_trajectory,
    )
    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_content_raw_scores_from_image",
        fail_image_sidecar,
    )

    cfg = _build_detect_cfg()
    cfg["__lf_formal_exact_context__"] = {
        "formal_exact_object_binding_status": "absent",
        "image_conditioned_reconstruction_available": False,
        "image_conditioned_reconstruction_status": "detect_input_image_absent",
    }

    record = detect_orchestrator.run_detect_orchestrator(
        cfg=cfg,
        impl_set=_build_impl_set(content_extractor),
        input_record=_build_input_record(attacked_image_path, "attacked_positive"),
        cfg_digest="cfg-digest",
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=None,
        detect_plan_result_override=_build_plan_override(),
    )

    dispatch_call = _extract_dispatch_call(content_extractor)
    detector_inputs = cast(Dict[str, Any], dispatch_call["inputs"])
    lf_trace = cast(Dict[str, Any], detector_inputs["lf_detect_trace"])
    assert lf_trace["lf_status"] == "absent"
    assert lf_trace["lf_absent_reason"] == "attack_image_conditioned_evidence_unavailable"
    assert lf_trace["lf_detect_path"] == "low_freq_template_image_conditioned_attack"
    assert cast(Dict[str, Any], record["content_evidence_payload"])["status"] == "absent"
    record_lf_trace = _extract_record_lf_trace(record)
    assert record_lf_trace["lf_absent_reason"] == "attack_image_conditioned_evidence_unavailable"


def test_run_detect_orchestrator_non_attacked_positive_keeps_formal_trajectory_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证非 attacked_positive 的 formal 主分发保持原有 trajectory 路径。

    Ensure non-attacked-positive paper-mode samples keep the existing formal
    trajectory dispatch path after the attacked-positive fix.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    clean_image_path = _write_test_image(tmp_path, "clean_main_dispatch.png")
    content_extractor = _CapturingContentExtractor()
    captures: List[Dict[str, Any]] = []

    def fake_extract_lf_raw_score_from_trajectory(**kwargs: Any) -> tuple[float, Dict[str, Any]]:
        captures.append(dict(kwargs))
        return 0.19, {
            "lf_status": "ok",
            "lf_detect_path": str(kwargs.get("detect_path", "low_freq_template_trajectory")),
        }

    def fail_image_sidecar(*args: Any, **kwargs: Any) -> tuple[float | None, float | None, Dict[str, Any]]:
        raise AssertionError("non-attacked-positive paper path must keep trajectory dispatch when sidecar is disabled")

    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        fake_extract_lf_raw_score_from_trajectory,
    )
    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_content_raw_scores_from_image",
        fail_image_sidecar,
    )

    record = detect_orchestrator.run_detect_orchestrator(
        cfg=_build_detect_cfg(),
        impl_set=_build_impl_set(content_extractor),
        input_record=_build_input_record(clean_image_path, "positive_source"),
        cfg_digest="cfg-digest",
        trajectory_evidence=None,
        injection_evidence=None,
        content_result_override=None,
        detect_plan_result_override=_build_plan_override(),
    )

    assert len(captures) == 1
    assert captures[0].get("latent_cache") is None
    assert captures[0].get("detect_path") is None
    dispatch_call = _extract_dispatch_call(content_extractor)
    detector_inputs = cast(Dict[str, Any], dispatch_call["inputs"])
    lf_trace = cast(Dict[str, Any], detector_inputs["lf_detect_trace"])
    assert lf_trace["lf_detect_path"] == "low_freq_template_trajectory"
    record_lf_trace = _extract_record_lf_trace(record)
    assert record_lf_trace["lf_detect_path"] == "low_freq_template_trajectory"