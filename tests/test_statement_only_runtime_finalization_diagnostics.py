"""
文件目的：验证 statement_only runtime finalization 的 planner 诊断透出与 formal failure artifact。
Module type: General module
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, cast

import numpy as np
from PIL import Image
import pytest

from main.cli import run_embed as run_embed_module
from main.core import digests
from main.diffusion.sd3 import infer_runtime as infer_runtime_module
from main.diffusion.sd3 import trajectory_tap as trajectory_tap_module
from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.watermarking.content_chain.subspace.planner_interface import SubspacePlanEvidence
from main.watermarking.content_chain.subspace.subspace_planner_impl import (
    SUBSPACE_PLANNER_ID,
    SUBSPACE_PLANNER_VERSION,
    SubspacePlannerImpl,
)


class _FakeContentExtractor:
    impl_version = "v1"

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        cfg_digest: str | None = None,
    ) -> Dict[str, Any]:
        _ = cfg
        _ = inputs
        _ = cfg_digest
        return {
            "status": "ok",
            "mask_digest": "mask_digest_anchor",
            "mask_stats": {
                "area_ratio": 0.25,
                "routing_digest": "routing_digest_anchor",
                "downsample_grid_true_indices": [0, 1, 2],
            },
        }


class _FakeSubspacePlanner:
    def __init__(self, planner_result: SubspacePlanEvidence) -> None:
        self._planner_result = planner_result
        self.calls: list[Dict[str, Any]] = []

    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: str | None = None,
        cfg_digest: str | None = None,
        inputs: Dict[str, Any] | None = None,
    ) -> SubspacePlanEvidence:
        self.calls.append(
            {
                "cfg": dict(cfg),
                "mask_digest": mask_digest,
                "cfg_digest": cfg_digest,
                "inputs": dict(inputs) if isinstance(inputs, dict) else inputs,
            }
        )
        return self._planner_result


class _FakeImplSet:
    def __init__(self, planner_result: SubspacePlanEvidence) -> None:
        self.content_extractor = _FakeContentExtractor()
        self.subspace_planner = _FakeSubspacePlanner(planner_result)


class _FakeImplIdentity:
    content_extractor_id = "unified_content_extractor"

    def as_dict(self) -> Dict[str, str]:
        return {"content_extractor_id": self.content_extractor_id}


class _FakePipelineWithDevice:
    def __init__(self, device_label: str, *, raise_on_move: str | None = None) -> None:
        self.device = device_label
        self.raise_on_move = raise_on_move
        self.to_calls: list[str] = []

    def to(self, device: Any) -> "_FakePipelineWithDevice":
        normalized_device = str(device)
        self.to_calls.append(normalized_device)
        if isinstance(self.raise_on_move, str) and self.raise_on_move:
            raise RuntimeError(self.raise_on_move)
        self.device = normalized_device
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        _ = kwargs
        return SimpleNamespace(images=[Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))])


def _build_planner_cfg() -> Dict[str, Any]:
    return {
        "paper_faithfulness": {"enabled": False},
        "watermark": {
            "subspace": {
                "enabled": True,
                "rank": 4,
                "sample_count": 4,
                "feature_dim": 8,
                "seed": 7,
                "jacobian_probe_count": 2,
                "jacobian_eps": 1e-3,
                "timestep_start": 0,
                "timestep_end": 3,
                "trajectory_step_stride": 1,
                "spectrum_topk": 4,
                "num_inference_steps": 28,
            }
        },
        "inference_num_steps": 28,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
    }


def _build_planner_inputs() -> Dict[str, Any]:
    feature_samples = np.arange(32, dtype=np.float64).reshape(4, 8)
    return {
        "trace_signature": {
            "num_inference_steps": 28,
            "guidance_scale": 7.0,
            "height": 512,
            "width": 512,
        },
        "mask_summary": {
            "area_ratio": 0.25,
            "routing_digest": "routing_digest_anchor",
            "downsample_grid_true_indices": [0, 1, 2],
        },
        "routing_digest": "routing_digest_anchor",
        "feature_samples": feature_samples,
    }


def _build_planner_failure_context() -> Dict[str, Any]:
    return {
        "diagnostic_version": "v1",
        "rank": 4,
        "feature_dim": 8,
        "sample_count": 4,
        "jvp_probe_count": 2,
        "trajectory_step_count": 28,
        "lf_route_count": 5,
        "hf_route_count": 3,
        "mask_summary_digest": "mask_summary_digest_anchor",
        "routing_digest": "routing_digest_anchor",
        "projection_matrix_shape": None,
        "trajectory_feature_matrix_shape": [4, 8],
        "jvp_matrix_shape": [4, 8],
        "has_nonfinite": False,
        "has_empty_partition": False,
        "has_empty_matrix": False,
        "has_dimension_mismatch": True,
        "stages": [
            {
                "stage_name": "routed_feature_selection_partition_build",
                "ok": True,
                "failure_reason_code": None,
                "exception_type": None,
                "exception_message": None,
                "lf_route_count": 5,
                "hf_route_count": 3,
                "trajectory_feature_matrix_shape": [4, 8],
            },
            {
                "stage_name": "routed_decomposition_matrix_build",
                "ok": False,
                "failure_reason_code": "routed_matrix_shape_mismatch",
                "exception_type": "ValueError",
                "exception_message": "synthetic routed matrix mismatch",
                "lf_route_count": 5,
                "hf_route_count": 3,
                "trajectory_feature_matrix_shape": [4, 8],
                "jvp_matrix_shape": [4, 8],
                "trajectory_jvp_feature_dim_mismatch": True,
            },
        ],
    }


def _build_runtime_capture_meta(
    status: str,
    *,
    step_count: int,
    failure_count: int,
    callback_invocation_count: int,
    callback_latent_present_count: int,
    available_steps: list[int] | None = None,
    missing_required_steps: list[int] | None = None,
    failure_examples: list[Dict[str, Any]] | None = None,
    attempt_count: int | None = None,
    success_count: int | None = None,
    required_step_count: int | None = None,
    tap_captured_step_count: int | None = None,
) -> Dict[str, Any]:
    normalized_available_steps = list(available_steps or [])
    normalized_missing_required_steps = list(missing_required_steps or [])
    normalized_failure_examples = [dict(item) for item in (failure_examples or [])]
    resolved_success_count = success_count if isinstance(success_count, int) else step_count
    resolved_attempt_count = attempt_count if isinstance(attempt_count, int) else resolved_success_count + failure_count
    resolved_required_step_count = required_step_count if isinstance(required_step_count, int) else len(normalized_available_steps) + len(normalized_missing_required_steps)
    resolved_tap_captured_step_count = tap_captured_step_count if isinstance(tap_captured_step_count, int) else callback_latent_present_count
    return {
        "trajectory_cache_capture_status": status,
        "trajectory_cache_step_count": step_count,
        "trajectory_cache_capture_attempt_count": resolved_attempt_count,
        "trajectory_cache_capture_success_count": resolved_success_count,
        "trajectory_cache_capture_failure_count": failure_count,
        "trajectory_cache_capture_failure_examples": normalized_failure_examples,
        "trajectory_cache_available_steps": normalized_available_steps,
        "trajectory_cache_required_step_count": resolved_required_step_count,
        "trajectory_cache_missing_required_steps": normalized_missing_required_steps,
        "trajectory_cache_callback_invocation_count": callback_invocation_count,
        "trajectory_cache_callback_latent_present_count": callback_latent_present_count,
        "trajectory_cache_tap_captured_step_count": resolved_tap_captured_step_count,
    }


def _build_trajectory_evidence(step_count: int) -> Dict[str, Any]:
    return {
        "status": "ok",
        "trajectory_metrics": {
            "steps": [
                {
                    "step_index": step_index,
                    "scheduler_step": step_index,
                    "scheduler_timestep": step_index,
                    "stats": {"mean": 0.0, "std": 1.0},
                }
                for step_index in range(step_count)
            ]
        },
    }


def _prepare_statement_only_failure_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    planner_result: SubspacePlanEvidence,
    runtime_capture_callable: Any | None = None,
    preview_generation_enabled: bool = False,
    runtime_device: str = "cpu",
) -> Dict[str, Any]:
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    captured: Dict[str, Any] = {"orchestrator_called": False, "runtime_capture_calls": 0}
    fake_impl_set = _FakeImplSet(planner_result)

    @contextmanager
    def _bound_fact_sources(*args: Any, **kwargs: Any) -> Iterator[None]:
        _ = args
        _ = kwargs
        yield

    def _ensure_output_layout(path: Path, **kwargs: Any) -> Dict[str, Path]:
        _ = kwargs
        records_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_root": path,
            "records_dir": records_dir,
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
        }

    def _fake_finalize_run(
        run_root_path: Path,
        records_dir_path: Path,
        artifacts_dir_path: Path,
        run_meta: Dict[str, Any],
    ) -> Path:
        _ = run_root_path
        _ = records_dir_path
        closure_payload: Dict[str, Any] = {
            "formal_two_stage": run_meta.get("formal_two_stage"),
            "status": {"details": run_meta.get("status_details")},
        }
        closure_path = artifacts_dir_path / "run_closure.json"
        closure_path.write_text(json.dumps(closure_payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        captured["run_closure_path"] = closure_path
        captured["formal_two_stage"] = run_meta.get("formal_two_stage")
        captured["status_details"] = run_meta.get("status_details")
        captured["preview_generation"] = run_meta.get("preview_generation")
        return closure_path

    def _load_and_validate_config(*args: Any, **kwargs: Any) -> tuple[Dict[str, Any], str, Dict[str, str]]:
        _ = args
        _ = kwargs
        return (
            {
                "policy_path": "content_np_geo_rescue",
                "paper_faithfulness": {"enabled": True},
                "attestation": {"enabled": True, "use_trajectory_mix": False},
                "detect": {"content": {"enabled": True}},
                "watermark": {
                    "hf": {"enabled": False},
                    "lf": {"enabled": True, "strength": 0.5},
                    "subspace": {
                        "enabled": True,
                        "rank": 4,
                        "sample_count": 4,
                        "feature_dim": 8,
                        "jacobian_probe_count": 2,
                        "jacobian_eps": 1e-3,
                    },
                },
                "inference_enabled": True,
                "inference_prompt": "statement only prompt",
                "inference_num_steps": 28,
                "inference_guidance_scale": 7.0,
                "inference_height": 512,
                "inference_width": 512,
                "device": runtime_device,
                "embed": {
                    "preview_generation": {
                        "enabled": preview_generation_enabled,
                        "artifact_rel_path": "preview/preview.png",
                    }
                },
            },
            "cfg_digest_anchor",
            {
                "cfg_pruned_for_digest_canon_sha256": "cfg_pruned_anchor",
                "cfg_audit_canon_sha256": "cfg_audit_anchor",
            },
        )

    def _fake_runtime_capture(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        captured["runtime_capture_calls"] += 1
        if callable(runtime_capture_callable):
            return cast(Dict[str, Any], runtime_capture_callable(*args, **kwargs))
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {"latency_ms": 1.0},
            "trajectory_evidence": {"status": "ok"},
            "injection_evidence": {},
            "trajectory_cache_capture_meta": _build_runtime_capture_meta(
                "all_failed",
                step_count=0,
                failure_count=28,
                callback_invocation_count=28,
                callback_latent_present_count=28,
                failure_examples=[
                    {
                        "step_index": 0,
                        "latent_type": "Tensor",
                        "exception_type": "RuntimeError",
                        "exception_message": "synthetic capture conversion failure",
                    }
                ],
            ),
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    def _fake_orchestrator(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        captured["orchestrator_called"] = True
        return {}

    def _derive_run_root(_output_dir: Any) -> Path:
        return run_root

    def _anchor_requirements(*args: Any) -> None:
        _ = args

    def _load_frozen_contracts(*args: Any) -> Dict[str, str]:
        _ = args
        return {"contracts": "ok"}

    def _load_runtime_whitelist(*args: Any) -> Dict[str, str]:
        _ = args
        return {"whitelist": "ok"}

    def _load_policy_path_semantics(*args: Any) -> Dict[str, str]:
        _ = args
        return {"semantics": "ok"}

    def _bind_freeze_anchors_to_run_meta(*args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs

    def _build_fact_sources_snapshot(*args: Any, **kwargs: Any) -> Dict[str, str]:
        _ = args
        _ = kwargs
        return {"snapshot": "ok"}

    def _assert_consistent_with_semantics(*args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs

    def _get_bound_fact_sources() -> Dict[str, str]:
        return {"snapshot": "ok"}

    def _get_contract_interpretation(*args: Any) -> SimpleNamespace:
        _ = args
        return SimpleNamespace()

    def _build_seed_audit(*args: Any, **kwargs: Any) -> tuple[Dict[str, Any], str, int, str]:
        _ = args
        _ = kwargs
        return ({}, "seed_digest", 7, "seed_rule")

    def _build_determinism_controls(*args: Any) -> None:
        _ = args

    def _normalize_nondeterminism_notes(*args: Any) -> None:
        _ = args

    def _build_pipeline_shell(*args: Any) -> Dict[str, Any]:
        _ = args
        return {
            "pipeline_obj": object(),
            "pipeline_provenance_canon_sha256": "pipeline_digest_anchor",
            "pipeline_status": "built",
            "pipeline_error": None,
            "pipeline_runtime_meta": {},
            "env_fingerprint_canon_sha256": "env_digest_anchor",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "model_digest_anchor",
        }

    def _build_runtime_impl_set_from_cfg(*args: Any) -> tuple[_FakeImplIdentity, _FakeImplSet, str]:
        _ = args
        return (_FakeImplIdentity(), fake_impl_set, "impl_cap_digest_anchor")

    def _compute_impl_identity_digest(*args: Any) -> str:
        _ = args
        return "impl_identity_digest_anchor"

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", _derive_run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.path_policy, "anchor_requirements", _anchor_requirements)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", _fake_finalize_run)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", _load_frozen_contracts)
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", _load_runtime_whitelist)
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", _load_policy_path_semantics)
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", cast(Any, lambda: {"manifest": "ok"}))
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", _bind_freeze_anchors_to_run_meta)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", _build_fact_sources_snapshot)
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", _assert_consistent_with_semantics)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", _get_bound_fact_sources)
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", _get_contract_interpretation)
    monkeypatch.setattr(run_embed_module.config_loader, "load_and_validate_config", _load_and_validate_config)
    monkeypatch.setattr(run_embed_module, "build_seed_audit", _build_seed_audit)
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", _build_determinism_controls)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", _normalize_nondeterminism_notes)
    monkeypatch.setattr(run_embed_module.pipeline_factory, "build_pipeline_shell", _build_pipeline_shell)
    monkeypatch.setattr(run_embed_module.runtime_resolver, "build_runtime_impl_set_from_cfg", _build_runtime_impl_set_from_cfg)
    monkeypatch.setattr(run_embed_module.runtime_resolver, "compute_impl_identity_digest", _compute_impl_identity_digest)
    monkeypatch.setattr(run_embed_module.infer_runtime, "run_sd3_inference", _fake_runtime_capture)
    monkeypatch.setattr(run_embed_module, "run_embed_orchestrator", _fake_orchestrator)

    captured["fake_impl_set"] = fake_impl_set
    return captured


def test_planner_failure_exposes_stage_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：planner 返回 invalid_subspace_params 时必须同时透出阶段、detail message 与诊断上下文。
    """
    planner = SubspacePlannerImpl(
        impl_id=SUBSPACE_PLANNER_ID,
        impl_version=SUBSPACE_PLANNER_VERSION,
        impl_digest=digests.canonical_sha256({"impl": "planner"}),
    )

    def _raise_routed_decomposition_failure(*args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        raise ValueError("synthetic routed decomposition failure")

    monkeypatch.setattr(planner, "_build_routed_decomposition_matrices", _raise_routed_decomposition_failure)

    result = planner.plan(
        _build_planner_cfg(),
        mask_digest="mask_digest_anchor",
        cfg_digest="cfg_digest_anchor",
        inputs=_build_planner_inputs(),
    )

    assert result.status == "failed"
    assert result.plan_failure_reason == "invalid_subspace_params"
    assert result.planner_failure_stage == "routed_decomposition_matrix_build"
    assert result.planner_failure_detail_message == "synthetic routed decomposition failure"
    assert isinstance(result.planner_diagnostic_context, dict)


def test_planner_failure_context_contains_shape_and_route_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：planner 失败诊断上下文必须包含 shape、route 与 rank 快照。
    """
    planner = SubspacePlannerImpl(
        impl_id=SUBSPACE_PLANNER_ID,
        impl_version=SUBSPACE_PLANNER_VERSION,
        impl_digest=digests.canonical_sha256({"impl": "planner"}),
    )

    def _raise_routed_decomposition_failure(*args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        raise ValueError("synthetic routed decomposition failure")

    monkeypatch.setattr(planner, "_build_routed_decomposition_matrices", _raise_routed_decomposition_failure)

    result = planner.plan(
        _build_planner_cfg(),
        mask_digest="mask_digest_anchor",
        cfg_digest="cfg_digest_anchor",
        inputs=_build_planner_inputs(),
    )

    context = cast(Dict[str, Any], result.planner_diagnostic_context)
    assert context["lf_route_count"] is not None
    assert context["hf_route_count"] is not None
    assert context["trajectory_feature_matrix_shape"] == [4, 8]
    assert context["jvp_matrix_shape"] is not None
    assert context["rank"] == 4


def test_run_embed_statement_only_failure_persists_runtime_finalization_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：statement_only runtime finalization 失败时必须把 planner 诊断写入 formal artifact 视图。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )
    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
    )

    with pytest.raises(
        ValueError,
        match="runtime executable formal plan unavailable: statement_only runtime finalization failed; reason=invalid_subspace_params",
    ):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    run_closure_path = cast(Path, env["run_closure_path"])
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))
    runtime_finalization = run_closure["status"]["details"]["runtime_finalization"]

    assert runtime_finalization["runtime_finalization_status"] == "failed"
    assert runtime_finalization["runtime_finalization_reason"] == "invalid_subspace_params"
    assert runtime_finalization["runtime_capture_inference_status"] == "ok"
    assert runtime_finalization["trajectory_cache_capture_status"] == "all_failed"
    assert runtime_finalization["trajectory_cache_step_count"] == 0
    assert runtime_finalization["trajectory_cache_capture_failure_count"] == 28
    assert runtime_finalization["trajectory_cache_capture_failure_examples"][0]["exception_message"] == "synthetic capture conversion failure"
    assert runtime_finalization["planner_failure_stage"] == "runtime_capture_cache_validation"
    assert runtime_finalization["planner_failure_detail_code"] == "trajectory_cache_capture_all_failed"
    assert runtime_finalization["planner_failure_detail_message"] == "trajectory_cache_capture_all_failed_cannot_build_basis"
    assert runtime_finalization["planner_diagnostic_context"]["trajectory_cache_capture"]["trajectory_cache_capture_status"] == "all_failed"


def test_trajectory_cache_capture_meta_reports_tap_observed_without_cache_write() -> None:
    """
    功能：当 tap 已观测到 step 但 cache 诊断仍为空壳时，必须返回显式不一致状态。
    """
    cache = LatentTrajectoryCache()

    capture_meta = trajectory_tap_module._build_trajectory_cache_capture_meta(
        cache,
        supports_callback=True,
        callback_invocation_count=28,
        callback_latent_present_count=28,
        tap_captured_step_count=28,
        required_cache_steps=list(range(28)),
    )

    assert isinstance(capture_meta, dict)
    assert capture_meta["trajectory_cache_capture_status"] == "tap_steps_observed_but_cache_write_not_observed"
    assert capture_meta["trajectory_cache_capture_attempt_count"] == 0
    assert capture_meta["trajectory_cache_capture_success_count"] == 0
    assert capture_meta["trajectory_cache_capture_failure_count"] == 0
    assert capture_meta["trajectory_cache_available_steps"] == []
    assert capture_meta["trajectory_cache_tap_captured_step_count"] == 28


def test_trajectory_cache_capture_meta_distinguishes_missing_latents_from_missing_callback() -> None:
    """
    功能：callback 已调用但 latents 缺失时，状态必须区别于完全未观察到 callback。
    """
    cache = LatentTrajectoryCache()

    callback_absent_meta = trajectory_tap_module._build_trajectory_cache_capture_meta(
        cache,
        supports_callback=True,
        callback_invocation_count=0,
        callback_latent_present_count=0,
        tap_captured_step_count=0,
        required_cache_steps=[0],
    )
    missing_latents_meta = trajectory_tap_module._build_trajectory_cache_capture_meta(
        cache,
        supports_callback=True,
        callback_invocation_count=3,
        callback_latent_present_count=0,
        tap_captured_step_count=0,
        required_cache_steps=[0],
    )

    assert isinstance(callback_absent_meta, dict)
    assert isinstance(missing_latents_meta, dict)
    assert callback_absent_meta["trajectory_cache_capture_status"] == "callback_not_observed"
    assert missing_latents_meta["trajectory_cache_capture_status"] == "callback_invoked_without_latents"
    assert callback_absent_meta["trajectory_cache_capture_status"] != missing_latents_meta["trajectory_cache_capture_status"]


def test_run_embed_statement_only_failure_reports_tap_observed_without_capture_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：当 runtime capture 失败且 meta 无法 reconstruction 时，formal failure 必须保留异常文本与精细失败编码。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )

    def _runtime_capture_missing_meta(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "inference_status": "failed",
            "inference_error": "synthetic_post_tap_failure",
            "inference_runtime_meta": {"latency_ms": 1.0},
            "trajectory_evidence": _build_trajectory_evidence(28),
            "injection_evidence": {},
            "trajectory_cache_capture_meta": None,
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        runtime_capture_callable=_runtime_capture_missing_meta,
    )

    with pytest.raises(
        ValueError,
        match="runtime executable formal plan unavailable: statement_only runtime finalization failed; reason=invalid_subspace_params",
    ):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    run_closure_path = cast(Path, env["run_closure_path"])
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))
    runtime_finalization = run_closure["status"]["details"]["runtime_finalization"]

    assert runtime_finalization["runtime_capture_inference_status"] == "failed"
    assert runtime_finalization["runtime_capture_inference_error"] == "synthetic_post_tap_failure"
    assert runtime_finalization["trajectory_cache_capture_status"] == "tap_steps_observed_but_cache_meta_missing"
    assert runtime_finalization["trajectory_cache_tap_captured_step_count"] == 28
    assert runtime_finalization["trajectory_cache_step_count"] == 0
    assert runtime_finalization["planner_failure_stage"] == "runtime_capture_cache_validation"
    assert runtime_finalization["planner_failure_detail_code"] == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure"
    assert runtime_finalization["planner_failure_detail_message"] == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure_cannot_build_basis"
    assert runtime_finalization["trajectory_cache_capture_error_message"] == "synthetic_post_tap_failure"


def test_run_sd3_inference_skips_redundant_pipeline_device_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：当 pipeline 已经位于目标设备时，不得再次执行重复 device move。
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")

    fake_pipeline = _FakePipelineWithDevice("cuda:0")

    def _fake_tap_from_pipeline(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "output": SimpleNamespace(images=[Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))]),
            "trajectory_evidence": {"status": "ok", "trajectory_metrics": {"steps": []}},
            "tap_status": "ok",
            "trajectory_cache_capture_meta": None,
        }

    monkeypatch.setattr(torch.cuda, "is_available", cast(Any, lambda: True))
    monkeypatch.setattr(infer_runtime_module.trajectory_tap, "tap_from_pipeline", _fake_tap_from_pipeline)

    result = infer_runtime_module.run_sd3_inference(
        {
            "inference_enabled": True,
            "inference_prompt": "prompt",
            "inference_num_steps": 4,
            "inference_guidance_scale": 7.0,
            "inference_height": 64,
            "inference_width": 64,
        },
        fake_pipeline,
        "cuda",
        None,
    )

    assert result["inference_status"] == "ok"
    assert fake_pipeline.to_calls == []
    assert result["inference_runtime_meta"]["pipeline_device_move_status"] == "skipped_already_on_target"
    assert result["inference_runtime_meta"]["pipeline_device_before_move"] == "cuda:0"
    assert result["inference_runtime_meta"]["pipeline_device_target"] == "cuda"


def test_run_sd3_inference_device_setup_error_returns_structured_capture_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：device setup 阶段失败时，必须返回结构化 capture 诊断而不是空壳结果。
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")

    fake_pipeline = _FakePipelineWithDevice("cpu", raise_on_move="CUDA out of memory")
    monkeypatch.setattr(torch.cuda, "is_available", cast(Any, lambda: True))

    result = infer_runtime_module.run_sd3_inference(
        {
            "inference_enabled": True,
            "inference_prompt": "prompt",
            "inference_num_steps": 4,
            "inference_guidance_scale": 7.0,
            "inference_height": 64,
            "inference_width": 64,
        },
        fake_pipeline,
        "cuda",
        None,
        trajectory_latent_cache=LatentTrajectoryCache(),
    )

    capture_meta = cast(Dict[str, Any], result["trajectory_cache_capture_meta"])
    assert result["inference_status"] == "failed"
    assert result["inference_error"] == "device_setup_error: CUDA out of memory"
    assert result["inference_runtime_meta"]["pipeline_device_move_status"] == "failed"
    assert capture_meta["trajectory_cache_capture_detail_code"] == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure"
    assert capture_meta["trajectory_cache_capture_error_message"] == "device_setup_error: CUDA out of memory"
    assert capture_meta["trajectory_cache_tap_captured_step_count"] == 0
    assert capture_meta["trajectory_cache_capture_status"] is None


def test_run_embed_statement_only_failure_persists_device_setup_error_before_tap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：当 second-pass 在 device setup 阶段失败时，formal artifact 必须保留精确异常且区别于 write_not_observed。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )

    def _runtime_capture_device_setup_error(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "inference_status": "failed",
            "inference_error": "device_setup_error: CUDA out of memory",
            "inference_runtime_meta": {
                "device": "cuda",
                "inference_status": "failed",
                "inference_error": "device_setup_error: CUDA out of memory",
                "pipeline_device_move_status": "failed",
            },
            "trajectory_evidence": {
                "status": "absent",
                "trajectory_absent_reason": "inference_failed",
            },
            "injection_evidence": {},
            "trajectory_cache_capture_meta": None,
        }

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        runtime_capture_callable=_runtime_capture_device_setup_error,
    )

    with pytest.raises(ValueError):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    run_closure_path = cast(Path, env["run_closure_path"])
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))
    runtime_finalization = run_closure["status"]["details"]["runtime_finalization"]

    assert runtime_finalization["runtime_capture_inference_status"] == "failed"
    assert runtime_finalization["runtime_capture_inference_error"] == "device_setup_error: CUDA out of memory"
    assert runtime_finalization["trajectory_cache_tap_captured_step_count"] == 0
    assert runtime_finalization["planner_failure_detail_code"] == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure"
    assert runtime_finalization["planner_failure_detail_message"] == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure_cannot_build_basis"
    assert runtime_finalization["planner_failure_detail_code"] != "trajectory_cache_write_not_observed_after_tap"


def test_run_embed_preview_generation_cleanup_preserves_statement_only_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：preview generation 结束后的最小清理不得破坏 preview 正式产物与 statement_only 成功路径。
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch unavailable")

    planner_result = SubspacePlanEvidence(
        status="ok",
        plan={"planner_params": {"rank": 4}},
        basis_digest="basis_digest_anchor",
        plan_digest="plan_digest_anchor",
        audit={
            "impl_identity": "subspace_planner",
            "impl_version": "v1",
            "impl_digest": "impl_digest_anchor",
            "trace_digest": "trace_digest_anchor",
        },
        plan_stats={"rank": 4},
        plan_failure_reason=None,
        planner_failure_stage=None,
        planner_failure_detail_code=None,
        planner_failure_detail_message=None,
        planner_diagnostic_context=None,
    )

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        preview_generation_enabled=True,
        runtime_device="cuda",
    )
    gc_collect_calls: list[str] = []
    empty_cache_calls: list[str] = []
    inference_call_count = 0

    def _runtime_capture_with_preview(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        nonlocal inference_call_count
        _ = args
        inference_call_count += 1
        call_index = inference_call_count
        cache = kwargs.get("trajectory_latent_cache")
        if call_index == 1:
            return {
                "inference_status": "ok",
                "inference_error": None,
                "inference_runtime_meta": {"latency_ms": 1.0},
                "trajectory_evidence": _build_trajectory_evidence(28),
                "injection_evidence": {},
                "trajectory_cache_capture_meta": None,
                "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
            }
        if isinstance(cache, LatentTrajectoryCache):
            cache.capture(0, np.zeros((1, 4, 8, 8), dtype=np.float32))
            cache.capture(1, np.ones((1, 4, 8, 8), dtype=np.float32))
            cache.capture(2, np.full((1, 4, 8, 8), 2.0, dtype=np.float32))
        if call_index == 2:
            return {
                "inference_status": "ok",
                "inference_error": None,
                "inference_runtime_meta": {"latency_ms": 1.0},
                "trajectory_evidence": {"status": "ok"},
                "injection_evidence": {},
                "trajectory_cache_capture_meta": _build_runtime_capture_meta(
                    "complete",
                    step_count=3,
                    failure_count=0,
                    callback_invocation_count=3,
                    callback_latent_present_count=3,
                    available_steps=[0, 1, 2],
                    required_step_count=3,
                ),
                "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
            }
        raise RuntimeError("stop_after_planner")

    monkeypatch.setattr(run_embed_module.gc, "collect", cast(Any, lambda: gc_collect_calls.append("collect") or 0))
    monkeypatch.setattr(torch.cuda, "is_available", cast(Any, lambda: True))
    monkeypatch.setattr(torch.cuda, "empty_cache", cast(Any, lambda: empty_cache_calls.append("empty_cache")))
    monkeypatch.setattr(run_embed_module.embed_orchestrator, "build_runtime_jvp_operator_from_cache", cast(Any, lambda cfg, cache: None))
    monkeypatch.setattr(run_embed_module.infer_runtime, "run_sd3_inference", _runtime_capture_with_preview)

    with pytest.raises(RuntimeError, match="stop_after_planner"):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    preview_generation = cast(Dict[str, Any], env["preview_generation"])
    assert preview_generation["status"] == "ok"
    assert preview_generation["persisted_artifact_rel_path"] == "preview/preview.png"
    assert gc_collect_calls == ["collect"]
    assert empty_cache_calls == ["empty_cache"]

    fake_impl_set = cast(_FakeImplSet, env["fake_impl_set"])
    assert len(fake_impl_set.subspace_planner.calls) == 1
    planner_call = fake_impl_set.subspace_planner.calls[0]
    cfg_snapshot = cast(Dict[str, Any], planner_call["cfg"])
    bound_cache = cfg_snapshot.get("__embed_trajectory_latent_cache__")
    assert isinstance(bound_cache, LatentTrajectoryCache)
    assert bound_cache.available_steps() == [0, 1, 2]


def test_run_embed_statement_only_failure_reports_tap_observed_meta_missing_without_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：当 tap 已捕获 step 但 meta 缺失且 runtime capture 未报错时，必须保留 after_tap 精细分支。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )

    def _runtime_capture_missing_meta(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {"latency_ms": 1.0},
            "trajectory_evidence": _build_trajectory_evidence(28),
            "injection_evidence": {},
            "trajectory_cache_capture_meta": None,
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        runtime_capture_callable=_runtime_capture_missing_meta,
    )

    with pytest.raises(ValueError):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    run_closure_path = cast(Path, env["run_closure_path"])
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))
    runtime_finalization = run_closure["status"]["details"]["runtime_finalization"]

    assert runtime_finalization["runtime_capture_inference_status"] == "ok"
    assert runtime_finalization["runtime_capture_inference_error"] is None
    assert runtime_finalization["trajectory_cache_capture_status"] == "tap_steps_observed_but_cache_meta_missing"
    assert runtime_finalization["planner_failure_detail_code"] == "trajectory_cache_capture_meta_missing_after_tap"
    assert runtime_finalization["planner_failure_detail_message"] == "trajectory_cache_capture_meta_missing_after_tap_cannot_build_basis"


def test_run_embed_statement_only_failure_preserves_write_not_observed_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：当 tap 已观测到 step 但 cache 未写入时，formal failure 必须继续落在 write_not_observed 分支。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )

    def _runtime_capture_write_not_observed(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {"latency_ms": 1.0},
            "trajectory_evidence": _build_trajectory_evidence(28),
            "injection_evidence": {},
            "trajectory_cache_capture_meta": _build_runtime_capture_meta(
                "tap_steps_observed_but_cache_write_not_observed",
                step_count=0,
                failure_count=0,
                callback_invocation_count=28,
                callback_latent_present_count=28,
                available_steps=[],
                tap_captured_step_count=28,
                attempt_count=0,
                success_count=0,
            ),
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        runtime_capture_callable=_runtime_capture_write_not_observed,
    )

    with pytest.raises(ValueError):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    run_closure_path = cast(Path, env["run_closure_path"])
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))
    runtime_finalization = run_closure["status"]["details"]["runtime_finalization"]

    assert runtime_finalization["trajectory_cache_capture_status"] == "tap_steps_observed_but_cache_write_not_observed"
    assert runtime_finalization["planner_failure_detail_code"] == "trajectory_cache_write_not_observed_after_tap"
    assert runtime_finalization["planner_failure_detail_message"] == "trajectory_cache_write_not_observed_after_tap_cannot_build_basis"


def test_run_embed_statement_only_failure_distinguishes_partial_cache_from_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：当仅有部分 available_steps 时，precheck detail 必须区别于空 cache。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )

    def _runtime_capture_partial_cache(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        cache = kwargs.get("trajectory_latent_cache")
        if isinstance(cache, LatentTrajectoryCache):
            cache.capture(0, np.zeros((1, 4, 8, 8), dtype=np.float32))
            cache.capture(1, np.ones((1, 4, 8, 8), dtype=np.float32))
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {"latency_ms": 1.0},
            "trajectory_evidence": _build_trajectory_evidence(4),
            "injection_evidence": {},
            "trajectory_cache_capture_meta": _build_runtime_capture_meta(
                "partial",
                step_count=2,
                failure_count=0,
                callback_invocation_count=4,
                callback_latent_present_count=4,
                available_steps=[0, 1],
                missing_required_steps=[2, 3],
                required_step_count=4,
                tap_captured_step_count=4,
                attempt_count=2,
                success_count=2,
            ),
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        runtime_capture_callable=_runtime_capture_partial_cache,
    )

    with pytest.raises(ValueError):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    run_closure_path = cast(Path, env["run_closure_path"])
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))
    runtime_finalization = run_closure["status"]["details"]["runtime_finalization"]

    assert runtime_finalization["trajectory_cache_capture_status"] == "partial"
    assert runtime_finalization["trajectory_cache_available_steps"] == [0, 1]
    assert runtime_finalization["trajectory_cache_missing_required_steps"] == [2, 3]
    assert runtime_finalization["planner_failure_detail_code"] == "trajectory_cache_partial_missing_required_steps"
    assert runtime_finalization["planner_failure_detail_message"] == (
        "trajectory_cache_partial_missing_required_steps_cannot_build_basis:[2, 3]"
    )


def test_run_embed_statement_only_failure_keeps_formal_gate_and_stops_before_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：runtime finalization 失败时主流程必须继续失败，且不能放松 formal gate 进入 orchestrator。
    """
    planner_result = SubspacePlanEvidence(
        status="failed",
        plan=None,
        basis_digest=None,
        plan_digest=None,
        audit=None,
        plan_stats=None,
        plan_failure_reason="invalid_subspace_params",
        planner_failure_stage="routed_decomposition_matrix_build",
        planner_failure_detail_code="routed_matrix_shape_mismatch",
        planner_failure_detail_message="synthetic routed matrix mismatch",
        planner_diagnostic_context=_build_planner_failure_context(),
    )
    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
    )

    with pytest.raises(ValueError):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    assert env["orchestrator_called"] is False
    assert env["runtime_capture_calls"] == 1
    fake_impl_set = cast(_FakeImplSet, env["fake_impl_set"])
    assert len(fake_impl_set.subspace_planner.calls) == 0


def test_run_embed_statement_only_success_binds_runtime_cache_into_planner_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：当 runtime trajectory cache 有真实 step 写入时，statement_only finalization 必须把 cache 绑定进 cfg 和 planner 输入。
    """
    planner_result = SubspacePlanEvidence(
        status="ok",
        plan={"planner_params": {"rank": 4}},
        basis_digest="basis_digest_anchor",
        plan_digest="plan_digest_anchor",
        audit={
            "impl_identity": "subspace_planner",
            "impl_version": "v1",
            "impl_digest": "impl_digest_anchor",
            "trace_digest": "trace_digest_anchor",
        },
        plan_stats={"rank": 4},
        plan_failure_reason=None,
        planner_failure_stage=None,
        planner_failure_detail_code=None,
        planner_failure_detail_message=None,
        planner_diagnostic_context=None,
    )

    def _runtime_capture_then_stop(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        cache = kwargs.get("trajectory_latent_cache")
        if isinstance(cache, LatentTrajectoryCache):
            cache.capture(0, np.zeros((1, 4, 8, 8), dtype=np.float32))
            cache.capture(1, np.ones((1, 4, 8, 8), dtype=np.float32))
            cache.capture(2, np.full((1, 4, 8, 8), 2.0, dtype=np.float32))
        if env["runtime_capture_calls"] == 1:
            return {
                "inference_status": "ok",
                "inference_error": None,
                "inference_runtime_meta": {"latency_ms": 1.0},
                "trajectory_evidence": {"status": "ok"},
                "injection_evidence": {},
                "trajectory_cache_capture_meta": _build_runtime_capture_meta(
                    "complete",
                    step_count=3,
                    failure_count=0,
                    callback_invocation_count=3,
                    callback_latent_present_count=3,
                    available_steps=[0, 1, 2],
                    required_step_count=3,
                ),
                "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
            }
        raise RuntimeError("stop_after_planner")

    env = _prepare_statement_only_failure_env(
        monkeypatch,
        tmp_path,
        planner_result=planner_result,
        runtime_capture_callable=_runtime_capture_then_stop,
    )
    monkeypatch.setattr(
        run_embed_module.embed_orchestrator,
        "build_runtime_jvp_operator_from_cache",
        cast(Any, lambda cfg, cache: None),
    )

    with pytest.raises(RuntimeError, match="stop_after_planner"):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    fake_impl_set = cast(_FakeImplSet, env["fake_impl_set"])
    assert len(fake_impl_set.subspace_planner.calls) == 1
    planner_call = fake_impl_set.subspace_planner.calls[0]
    cfg_snapshot = cast(Dict[str, Any], planner_call["cfg"])
    bound_cache = cfg_snapshot.get("__embed_trajectory_latent_cache__")
    assert isinstance(bound_cache, LatentTrajectoryCache)
    assert bound_cache.available_steps() == [0, 1, 2]
    planner_inputs = cast(Dict[str, Any], planner_call["inputs"])
    runtime_cache_input = planner_inputs.get("trajectory_latent_cache")
    assert isinstance(runtime_cache_input, LatentTrajectoryCache)
    assert runtime_cache_input.available_steps() == [0, 1, 2]


def test_runtime_capture_precheck_failure_codes_remain_distinguishable() -> None:
    """
    功能：callback、all_failed、partial、meta_missing、write_not_observed 等分支必须保持可区分。
    """
    diagnostics_cases = {
        "callback_not_observed": {
            "trajectory_cache_capture_status": "callback_not_observed",
            "trajectory_cache_capture_detail_code": "trajectory_callback_not_observed",
            "trajectory_cache_available_steps": [],
            "trajectory_cache_missing_required_steps": [],
        },
        "callback_invoked_without_latents": {
            "trajectory_cache_capture_status": "callback_invoked_without_latents",
            "trajectory_cache_capture_detail_code": "trajectory_callback_invoked_without_latents",
            "trajectory_cache_available_steps": [],
            "trajectory_cache_missing_required_steps": [],
        },
        "all_failed": {
            "trajectory_cache_capture_status": "all_failed",
            "trajectory_cache_capture_detail_code": "trajectory_cache_capture_all_failed",
            "trajectory_cache_available_steps": [],
            "trajectory_cache_missing_required_steps": [],
        },
        "partial": {
            "trajectory_cache_capture_status": "partial",
            "trajectory_cache_capture_detail_code": "trajectory_cache_partial_missing_required_steps",
            "trajectory_cache_available_steps": [0, 1],
            "trajectory_cache_missing_required_steps": [2],
        },
        "meta_missing": {
            "trajectory_cache_capture_status": "tap_steps_observed_but_cache_meta_missing",
            "trajectory_cache_capture_detail_code": "trajectory_cache_capture_meta_missing_after_tap",
            "trajectory_cache_available_steps": [],
            "trajectory_cache_missing_required_steps": [],
        },
        "write_not_observed": {
            "trajectory_cache_capture_status": "tap_steps_observed_but_cache_write_not_observed",
            "trajectory_cache_capture_detail_code": "trajectory_cache_write_not_observed_after_tap",
            "trajectory_cache_available_steps": [],
            "trajectory_cache_missing_required_steps": [],
        },
        "meta_unreconstructable_after_runtime_failure": {
            "trajectory_cache_capture_status": None,
            "trajectory_cache_capture_detail_code": "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure",
            "trajectory_cache_available_steps": [],
            "trajectory_cache_missing_required_steps": [],
            "trajectory_cache_tap_captured_step_count": 0,
        },
    }

    detail_codes = {
        case_name: run_embed_module._build_runtime_capture_precheck_failure(diagnostics)["planner_failure_detail_code"]
        for case_name, diagnostics in diagnostics_cases.items()
    }

    assert detail_codes["callback_not_observed"] == "trajectory_callback_not_observed"
    assert detail_codes["callback_invoked_without_latents"] == "trajectory_callback_invoked_without_latents"
    assert detail_codes["all_failed"] == "trajectory_cache_capture_all_failed"
    assert detail_codes["partial"] == "trajectory_cache_partial_missing_required_steps"
    assert detail_codes["meta_missing"] == "trajectory_cache_capture_meta_missing_after_tap"
    assert detail_codes["write_not_observed"] == "trajectory_cache_write_not_observed_after_tap"
    assert detail_codes["meta_unreconstructable_after_runtime_failure"] == "trajectory_cache_capture_meta_unreconstructable_after_runtime_failure"
    assert len(set(detail_codes.values())) == len(detail_codes)


def test_latent_trajectory_cache_capture_reports_single_step_failure() -> None:
    """
    功能：单步 latent 转换失败时，LatentTrajectoryCache 必须保留失败计数与失败样例。
    """

    class _BrokenLatent:
        def detach(self) -> "_BrokenLatent":
            return self

        def float(self) -> "_BrokenLatent":
            return self

        def cpu(self) -> "_BrokenLatent":
            return self

        def numpy(self) -> Any:
            raise RuntimeError("broken_numpy_conversion")

    cache = LatentTrajectoryCache()
    cache.capture(0, _BrokenLatent())

    diagnostics = cache.capture_diagnostics()
    assert diagnostics["capture_attempt_count"] == 1
    assert diagnostics["capture_success_count"] == 0
    assert diagnostics["capture_failure_count"] == 1
    assert diagnostics["available_steps"] == []
    assert diagnostics["failure_examples"][0]["step_index"] == 0
    assert diagnostics["failure_examples"][0]["exception_message"] == "broken_numpy_conversion"
