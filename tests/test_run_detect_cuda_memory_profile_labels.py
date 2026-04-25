"""
文件目的：验证 run_detect 在主推理与 same_seed_control rerun 上显式传入稳定的 cuda memory phase label。
Module type: General module
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, cast

import pytest

from main.cli import run_detect as run_detect_module


class StopAfterDetectInference(Exception):
    """Sentinel exception used to stop run_detect at the main inference call."""


class _FakeDetectContentExtractor:
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
        return {"status": "ok", "mask_digest": "mask_digest_anchor"}


class _FakeDetectPlanResult:
    def __init__(self) -> None:
        self.plan_digest = "plan_digest_anchor"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "plan_digest": "plan_digest_anchor",
            "basis_digest": "basis_digest_anchor",
        }


class _FakeDetectSubspacePlanner:
    def plan(
        self,
        cfg: Dict[str, Any],
        mask_digest: str | None = None,
        cfg_digest: str | None = None,
        inputs: Dict[str, Any] | None = None,
    ) -> _FakeDetectPlanResult:
        _ = cfg
        _ = mask_digest
        _ = cfg_digest
        _ = inputs
        return _FakeDetectPlanResult()


class _FakeDetectImplSet:
    def __init__(self) -> None:
        self.content_extractor = _FakeDetectContentExtractor()
        self.subspace_planner = _FakeDetectSubspacePlanner()


class _FakeDetectImplIdentity:
    content_extractor_id = "unified_content_extractor"

    def as_dict(self) -> Dict[str, str]:
        return {"content_extractor_id": self.content_extractor_id}


def _prepare_detect_main_inference_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    captured_kwargs: Dict[str, Any],
) -> None:
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    @contextmanager
    def _bound_fact_sources(*args: Any, **kwargs: Any) -> Iterator[None]:
        _ = args
        _ = kwargs
        yield

    def _derive_run_root(_output_dir: Any) -> Path:
        return run_root

    def _ensure_output_layout(path: Path, **kwargs: Any) -> Dict[str, Path]:
        _ = path
        _ = kwargs
        records_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_root": run_root,
            "records_dir": records_dir,
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
        }

    def _fake_inference(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        captured_kwargs.update(kwargs)
        raise StopAfterDetectInference("stop after detect main inference")

    def _anchor_requirements(*args: Any) -> None:
        _ = args

    def _finalize_run(*args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs

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

    def _get_contract_interpretation(*args: Any) -> SimpleNamespace:
        _ = args
        return SimpleNamespace()

    def _load_and_validate_config(*args: Any, **kwargs: Any) -> tuple[Dict[str, Any], str, Dict[str, str]]:
        _ = args
        _ = kwargs
        return (
            {
                "policy_path": "content_np_geo_rescue",
                "detect": {
                    "content": {"enabled": True},
                    "geometry": {"enabled": False, "enable_attention_anchor": False},
                },
                "watermark": {"hf": {"enabled": False}},
                "inference_enabled": True,
                "inference_prompt": "detect prompt",
                "inference_num_steps": 4,
                "inference_guidance_scale": 7.0,
                "inference_height": 64,
                "inference_width": 64,
                "device": "cpu",
            },
            "cfg_digest_anchor",
            {
                "cfg_pruned_for_digest_canon_sha256": "cfg_pruned_anchor",
                "cfg_audit_canon_sha256": "cfg_audit_anchor",
            },
        )

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

    def _build_runtime_impl_set_from_cfg(*args: Any) -> tuple[_FakeDetectImplIdentity, _FakeDetectImplSet, str]:
        _ = args
        return (_FakeDetectImplIdentity(), _FakeDetectImplSet(), "impl_cap_digest_anchor")

    def _compute_impl_identity_digest(*args: Any) -> str:
        _ = args
        return "impl_identity_digest_anchor"

    monkeypatch.setattr(run_detect_module.path_policy, "derive_run_root", _derive_run_root)
    monkeypatch.setattr(run_detect_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_detect_module.path_policy, "anchor_requirements", _anchor_requirements)
    monkeypatch.setattr(run_detect_module.status, "finalize_run", _finalize_run)
    monkeypatch.setattr(run_detect_module, "load_frozen_contracts", _load_frozen_contracts)
    monkeypatch.setattr(run_detect_module, "load_runtime_whitelist", _load_runtime_whitelist)
    monkeypatch.setattr(run_detect_module, "load_policy_path_semantics", _load_policy_path_semantics)
    monkeypatch.setattr(run_detect_module.config_loader, "load_injection_scope_manifest", cast(Any, lambda: {"manifest": "ok"}))
    monkeypatch.setattr(run_detect_module.status, "bind_freeze_anchors_to_run_meta", _bind_freeze_anchors_to_run_meta)
    monkeypatch.setattr(run_detect_module.records_io, "build_fact_sources_snapshot", _build_fact_sources_snapshot)
    monkeypatch.setattr(run_detect_module, "assert_consistent_with_semantics", _assert_consistent_with_semantics)
    monkeypatch.setattr(run_detect_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_detect_module.records_io, "get_bound_fact_sources", cast(Any, lambda: {"snapshot": "ok"}))
    monkeypatch.setattr(run_detect_module, "get_contract_interpretation", _get_contract_interpretation)
    monkeypatch.setattr(run_detect_module.config_loader, "load_and_validate_config", _load_and_validate_config)
    monkeypatch.setattr(run_detect_module, "build_seed_audit", _build_seed_audit)
    monkeypatch.setattr(run_detect_module, "build_determinism_controls", _build_determinism_controls)
    monkeypatch.setattr(run_detect_module, "normalize_nondeterminism_notes", _normalize_nondeterminism_notes)
    monkeypatch.setattr(run_detect_module.pipeline_factory, "build_pipeline_shell", _build_pipeline_shell)
    monkeypatch.setattr(run_detect_module.runtime_resolver, "build_runtime_impl_set_from_cfg", _build_runtime_impl_set_from_cfg)
    monkeypatch.setattr(run_detect_module.runtime_resolver, "compute_impl_identity_digest", _compute_impl_identity_digest)
    monkeypatch.setattr(run_detect_module, "assert_detect_runtime_dependencies", cast(Any, lambda *args, **kwargs: {"allow_flag": False}))
    monkeypatch.setattr(run_detect_module, "build_injection_context_from_plan", cast(Any, lambda *args, **kwargs: object()))
    monkeypatch.setattr(run_detect_module, "_build_planner_inputs_for_runtime", cast(Any, lambda *args, **kwargs: {}))
    monkeypatch.setattr(run_detect_module.infer_runtime, "run_sd3_inference", _fake_inference)


def test_run_detect_main_inference_passes_runtime_phase_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：detect 主推理必须显式传入固定的 runtime_phase_label。
    """
    captured_kwargs: Dict[str, Any] = {}
    _prepare_detect_main_inference_env(monkeypatch, tmp_path, captured_kwargs)

    with pytest.raises(StopAfterDetectInference):
        run_detect_module.run_detect(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            input_record_path=None,
            overrides=None,
            thresholds_path=None,
        )

    assert captured_kwargs["runtime_phase_label"] == run_detect_module.DETECT_MAIN_INFERENCE_PHASE_LABEL


def test_same_seed_control_rerun_passes_runtime_phase_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：same_seed_control rerun 必须显式传入固定的 runtime_phase_label。
    """
    captured_kwargs: Dict[str, Any] = {}

    def _fake_inference(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        captured_kwargs.update(kwargs)
        return {
            "inference_status": "failed",
            "inference_error": "synthetic_same_seed_stop",
            "trajectory_evidence": {},
        }

    monkeypatch.setattr(run_detect_module.infer_runtime, "run_sd3_inference", _fake_inference)

    context = run_detect_module._build_lf_protocol_control_context(  # pyright: ignore[reportPrivateUsage]
        {"detect": {"content": {"enabled": True}}},
        {"seed_value": 7},
        {"plan_digest": "plan_digest_anchor"},
        object(),
        "cpu",
        detect_seed=8,
        injection_context=None,
        injection_modifier=None,
    )

    assert captured_kwargs["runtime_phase_label"] == run_detect_module.DETECT_SAME_SEED_CONTROL_INFERENCE_PHASE_LABEL
    assert context["same_seed_control_reason"] == "same_seed_control_inference_unavailable"


def test_run_detect_runtime_session_reuses_static_runtime_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：runtime_session 复用时必须保留 event 级 config 加载，并跳过重复的 pipeline / impl set 构建。
    """
    captured_kwargs: Dict[str, Any] = {}
    call_counts = {
        "load_and_validate_config": 0,
        "build_pipeline_shell": 0,
        "build_runtime_impl_set_from_cfg": 0,
    }

    _prepare_detect_main_inference_env(monkeypatch, tmp_path, captured_kwargs)

    original_load_and_validate_config = run_detect_module.config_loader.load_and_validate_config
    original_build_pipeline_shell = run_detect_module.pipeline_factory.build_pipeline_shell
    original_build_runtime_impl_set_from_cfg = run_detect_module.runtime_resolver.build_runtime_impl_set_from_cfg

    def _counted_load_and_validate_config(*args: Any, **kwargs: Any) -> tuple[Dict[str, Any], str, Dict[str, str]]:
        call_counts["load_and_validate_config"] += 1
        return original_load_and_validate_config(*args, **kwargs)

    def _counted_build_pipeline_shell(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        call_counts["build_pipeline_shell"] += 1
        return original_build_pipeline_shell(*args, **kwargs)

    def _counted_build_runtime_impl_set_from_cfg(*args: Any, **kwargs: Any) -> tuple[_FakeDetectImplIdentity, _FakeDetectImplSet, str]:
        call_counts["build_runtime_impl_set_from_cfg"] += 1
        return original_build_runtime_impl_set_from_cfg(*args, **kwargs)

    monkeypatch.setattr(run_detect_module.config_loader, "load_and_validate_config", _counted_load_and_validate_config)
    monkeypatch.setattr(run_detect_module.pipeline_factory, "build_pipeline_shell", _counted_build_pipeline_shell)
    monkeypatch.setattr(
        run_detect_module.runtime_resolver,
        "build_runtime_impl_set_from_cfg",
        _counted_build_runtime_impl_set_from_cfg,
    )

    runtime_session = run_detect_module.build_detect_runtime_session("configs/default.yaml")

    assert call_counts["load_and_validate_config"] == 1
    assert call_counts["build_pipeline_shell"] == 1
    assert call_counts["build_runtime_impl_set_from_cfg"] == 1

    with pytest.raises(StopAfterDetectInference):
        run_detect_module.run_detect(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            input_record_path=None,
            overrides=None,
            thresholds_path=None,
            runtime_session=runtime_session,
        )

    assert call_counts["load_and_validate_config"] == 2
    assert call_counts["build_pipeline_shell"] == 1
    assert call_counts["build_runtime_impl_set_from_cfg"] == 1
    assert captured_kwargs["runtime_phase_label"] == run_detect_module.DETECT_MAIN_INFERENCE_PHASE_LABEL