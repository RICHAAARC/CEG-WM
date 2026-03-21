"""
File purpose: run_embed CLI 输入绑定顺序回归测试。
Module type: General module
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import numpy as np

from main.cli import run_embed as run_embed_module


class _StopAfterContentExtract(Exception):
    """Sentinel exception to stop run_embed after precompute extract."""


class _StopAfterPlannerPrecompute(Exception):
    """Sentinel exception to stop run_embed after planner precompute."""


class _StopAfterRunEmbedOrchestrator(Exception):
    """Sentinel exception to stop run_embed after entering embed orchestrator."""


class _FakeSubspacePlanResult:
    """Minimal subspace result payload for run_embed precompute tests."""

    def __init__(self, status: str, plan: dict | None, plan_digest: str | None, basis_digest: str | None, plan_failure_reason: str | None = None) -> None:
        self.status = status
        self.plan = plan
        self.plan_digest = plan_digest
        self.basis_digest = basis_digest
        self.plan_failure_reason = plan_failure_reason

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "plan": self.plan,
            "plan_digest": self.plan_digest,
            "basis_digest": self.basis_digest,
            "plan_failure_reason": self.plan_failure_reason,
            "plan_stats": {
                "planner_status": self.status,
                "planner_absent_reason": self.plan_failure_reason,
            },
        }


def test_bind_embed_plan_digest_consistency_marks_match() -> None:
    """
    功能：一致的 injection/formal plan digest 必须写出 match 语义。 

    Verify matching injection/formal plan digests emit match semantics.

    Args:
        None.

    Returns:
        None.
    """
    record = {}
    content_evidence = {"status": "ok"}

    status = run_embed_module._bind_embed_plan_digest_consistency(
        record,
        content_evidence,
        "plan_digest_anchor",
        "plan_digest_anchor",
    )

    assert status == "ok"
    assert record["plan_digest_injection"] == "plan_digest_anchor"
    assert record["plan_digest_formal"] == "plan_digest_anchor"
    assert record["plan_digest_expected"] == "plan_digest_anchor"
    assert record["plan_digest_observed"] == "plan_digest_anchor"
    assert record["plan_digest_status"] == "ok"
    assert record["plan_digest_match_status"] == "match"
    assert "plan_digest_mismatch_reason" not in record
    assert content_evidence["status"] == "ok"
    assert "plan_digest_mismatch" not in content_evidence


def test_bind_embed_plan_digest_consistency_marks_mismatch() -> None:
    """
    功能：分叉的 injection/formal plan digest 必须写出 mismatch 语义。 

    Verify divergent injection/formal plan digests emit mismatch semantics.

    Args:
        None.

    Returns:
        None.
    """
    record = {}
    content_evidence = {"status": "ok"}

    status = run_embed_module._bind_embed_plan_digest_consistency(
        record,
        content_evidence,
        "plan_digest_injection_anchor",
        "plan_digest_formal_anchor",
    )

    assert status == "mismatch"
    assert record["plan_digest_injection"] == "plan_digest_injection_anchor"
    assert record["plan_digest_formal"] == "plan_digest_formal_anchor"
    assert record["plan_digest_expected"] == "plan_digest_injection_anchor"
    assert record["plan_digest_observed"] == "plan_digest_formal_anchor"
    assert record["plan_digest_status"] == "mismatch"
    assert record["plan_digest_match_status"] == "mismatch"
    assert record["plan_digest_mismatch_reason"] == "plan_digest_mismatch"
    assert content_evidence["status"] == "mismatch"
    assert content_evidence["content_mismatch_reason"] == "plan_digest_mismatch"
    assert content_evidence["plan_digest_mismatch"] == {
        "plan_digest_injection": "plan_digest_injection_anchor",
        "plan_digest_formal": "plan_digest_formal_anchor",
        "mismatch_reason": "plan_digest_mismatch",
    }


def test_bind_embed_plan_digest_consistency_marks_formal_absent_for_compatibility_path_only() -> None:
    """
    功能：兼容路径若已发生 fallback 注入，不得误报为 injection digest 缺失。 

    Verify fallback injection keeps the injection digest populated and reports
    a formal-absent mismatch instead of an injection-absent mismatch.
    This helper-level behavior is reserved for compatibility paths and must not
    be interpreted as legal formal-path behavior.

    Args:
        None.

    Returns:
        None.
    """
    record = {}
    content_evidence = {"status": "ok", "fallback_plan_digest": "fallback_digest_anchor"}

    status = run_embed_module._bind_embed_plan_digest_consistency(
        record,
        content_evidence,
        "fallback_digest_anchor",
        None,
    )

    assert status == "absent"
    assert record["plan_digest_injection"] == "fallback_digest_anchor"
    assert record["plan_digest_formal"] == "<absent>"
    assert record["plan_digest_match_status"] == "absent"
    assert record["plan_digest_mismatch_reason"] == "plan_digest_formal_absent"
    assert content_evidence["content_mismatch_reason"] == "plan_digest_formal_absent"


def test_resolve_formal_subspace_override_requires_plan_and_basis() -> None:
    """
    功能：仅当 precomputed formal plan 与 basis 都存在时才允许复用 override。 

    Verify the orchestrator only reuses a precomputed subspace override when
    both formal plan and basis digests are available.

    Args:
        None.

    Returns:
        None.
    """
    absent_precompute = {
        "status": "absent",
        "plan": {},
        "plan_digest": None,
        "basis_digest": None,
        "plan_stats": {"planner_status": "absent", "planner_absent_reason": "mask_absent"},
    }
    formal_precompute = {
        "status": "ok",
        "plan": {"rank": 8},
        "plan_digest": "formal_plan_anchor",
        "basis_digest": "formal_basis_anchor",
    }

    assert run_embed_module._resolve_formal_subspace_override(absent_precompute) is None
    assert run_embed_module._resolve_formal_subspace_override(formal_precompute) is formal_precompute


def test_build_statement_only_formal_scaffold_uses_static_input_domain() -> None:
    """
    功能：statement_only formal scaffold 只能绑定 pre-inference 静态输入域。
    """
    cfg_payload = {
        "policy_path": "content_np_geo_rescue",
        "model_id": "sd3-test",
        "inference_prompt": "scaffold prompt",
    }
    content_payload = {
        "mask_digest": "mask_digest_anchor",
        "mask_stats": {"routing_digest": "routing_digest_anchor", "area_ratio": 0.25},
    }
    planner_inputs = {
        "trace_signature": {"num_inference_steps": 4, "guidance_scale": 7.0, "height": 512, "width": 512},
        "mask_summary": {"routing_digest": "routing_digest_anchor", "area_ratio": 0.25},
        "routing_digest": "routing_digest_anchor",
        "jvp_operator": object(),
    }

    scaffold, failure_reason = run_embed_module._build_statement_only_formal_scaffold(
        cfg_payload,
        "cfg_digest_anchor",
        7,
        content_payload,
        planner_inputs,
    )

    assert failure_reason is None
    assert isinstance(scaffold, dict)
    assert scaffold["formal_object_stage"] == "pre_inference_scaffold"
    assert scaffold["mask_digest"] == "mask_digest_anchor"
    assert scaffold["routing_digest"] == "routing_digest_anchor"
    assert isinstance(scaffold.get("formal_scaffold_digest"), str)
    assert "plan_digest" not in scaffold
    assert "basis_digest" not in scaffold


def test_build_statement_only_formal_scaffold_requires_routing_digest() -> None:
    """
    功能：statement_only formal scaffold 缺少 routing_digest 时必须失败。
    """
    scaffold, failure_reason = run_embed_module._build_statement_only_formal_scaffold(
        {"policy_path": "content_np_geo_rescue"},
        "cfg_digest_anchor",
        7,
        {
            "mask_digest": "mask_digest_anchor",
            "mask_stats": {"area_ratio": 0.25},
        },
        {
            "trace_signature": {"num_inference_steps": 4},
            "mask_summary": {"area_ratio": 0.25},
        },
    )

    assert scaffold is None
    assert failure_reason == "scaffold_routing_digest_absent"


def test_run_embed_binds_input_image_before_content_precompute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 run_embed 在 content 预计算前绑定 __embed_input_image_path__。

    Verify run_embed binds embed input image path before content precompute.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    input_image = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(input_image)

    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    @contextmanager
    def _bound_fact_sources(*args, **kwargs):
        _ = args
        _ = kwargs
        yield

    def _ensure_output_layout(path: Path, **kwargs):
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

    cfg_payload = {
        "policy_path": "content_only",
        "embed": {},
        "watermark": {"hf": {"enabled": False}},
    }

    captured_inputs = {}

    class _FakeContentExtractor:
        impl_version = "v1"

        def extract(self, cfg, inputs=None, cfg_digest=None):
            _ = cfg
            _ = cfg_digest
            captured_inputs["inputs"] = inputs
            raise _StopAfterContentExtract("stop after precompute extract")

    class _FakeImplSet:
        def __init__(self):
            self.content_extractor = _FakeContentExtractor()

    class _FakeImplIdentity:
        content_extractor_id = "unified_content_extractor_v1"

        def as_dict(self):
            return {"content_extractor_id": self.content_extractor_id}

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", lambda _: run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", lambda *_: {"contracts": "ok"})
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", lambda *_: {"whitelist": "ok"})
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", lambda *_: {"semantics": "ok"})
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", lambda: {"manifest": "ok"})
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", lambda *args, **kwargs: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", lambda: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", lambda *_: SimpleNamespace())
    monkeypatch.setattr(
        run_embed_module.config_loader,
        "load_and_validate_config",
        lambda *args, **kwargs: (dict(cfg_payload), "cfg_digest_anchor", {"cfg_pruned_for_digest_canon_sha256": "a", "cfg_audit_canon_sha256": "b"}),
    )
    monkeypatch.setattr(run_embed_module, "build_seed_audit", lambda *args, **kwargs: ({}, "seed_digest", 42, "seed_rule"))
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", lambda *_: None)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", lambda *_: None)
    monkeypatch.setattr(
        run_embed_module.pipeline_factory,
        "build_pipeline_shell",
        lambda *_: {
            "pipeline_provenance_canon_sha256": "<absent>",
            "pipeline_status": "unbuilt",
            "pipeline_error": "<absent>",
            "pipeline_runtime_meta": None,
            "env_fingerprint_canon_sha256": "<absent>",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "<absent>",
        },
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "build_runtime_impl_set_from_cfg",
        lambda *_: (_FakeImplIdentity(), _FakeImplSet(), "impl_cap_digest"),
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "compute_impl_identity_digest",
        lambda *_: "impl_identity_digest_anchor",
    )

    with pytest.raises(_StopAfterContentExtract):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=str(input_image),
        )

    assert isinstance(captured_inputs.get("inputs"), dict)
    assert captured_inputs["inputs"].get("image_path") == str(input_image)


def test_run_embed_preview_generation_binds_generated_input_before_content_precompute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：preview_generation 必须读取 inference_status 并将预览图绑定到 content 预计算输入。

    Verify run_embed consumes inference_status from run_sd3_inference and binds
    the generated preview image into content precompute inputs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    @contextmanager
    def _bound_fact_sources(*args, **kwargs):
        _ = args
        _ = kwargs
        yield

    def _ensure_output_layout(path: Path, **kwargs):
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

    cfg_payload = {
        "policy_path": "content_only",
        "embed": {
            "preview_generation": {
                "enabled": True,
            },
        },
        "detect": {
            "content": {
                "enabled": True,
            },
        },
        "watermark": {"hf": {"enabled": False}},
        "inference_enabled": True,
        "inference_prompt": "preview prompt",
        "inference_num_steps": 1,
        "inference_guidance_scale": 1.0,
        "inference_height": 8,
        "inference_width": 8,
        "device": "cpu",
    }

    captured_inputs = {}

    class _FakeContentExtractor:
        impl_version = "v1"

        def extract(self, cfg, inputs=None, cfg_digest=None):
            _ = cfg
            _ = cfg_digest
            captured_inputs["inputs"] = inputs
            raise _StopAfterContentExtract("stop after preview content precompute")

    class _FakeImplSet:
        def __init__(self):
            self.content_extractor = _FakeContentExtractor()

    class _FakeImplIdentity:
        content_extractor_id = "unified_content_extractor_v1"

        def as_dict(self):
            return {"content_extractor_id": self.content_extractor_id}

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", lambda _: run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", lambda *_: {"contracts": "ok"})
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", lambda *_: {"whitelist": "ok"})
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", lambda *_: {"semantics": "ok"})
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", lambda: {"manifest": "ok"})
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", lambda *args, **kwargs: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", lambda: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", lambda *_: SimpleNamespace())
    monkeypatch.setattr(
        run_embed_module.config_loader,
        "load_and_validate_config",
        lambda *args, **kwargs: (dict(cfg_payload), "cfg_digest_anchor", {"cfg_pruned_for_digest_canon_sha256": "a", "cfg_audit_canon_sha256": "b"}),
    )
    monkeypatch.setattr(run_embed_module, "build_seed_audit", lambda *args, **kwargs: ({}, "seed_digest", 7, "seed_rule"))
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", lambda *_: None)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", lambda *_: None)
    monkeypatch.setattr(
        run_embed_module.pipeline_factory,
        "build_pipeline_shell",
        lambda *_: {
            "pipeline_obj": object(),
            "pipeline_provenance_canon_sha256": "<absent>",
            "pipeline_status": "built",
            "pipeline_error": "<absent>",
            "pipeline_runtime_meta": {},
            "env_fingerprint_canon_sha256": "<absent>",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "<absent>",
        },
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "build_runtime_impl_set_from_cfg",
        lambda *_: (_FakeImplIdentity(), _FakeImplSet(), "impl_cap_digest"),
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "compute_impl_identity_digest",
        lambda *_: "impl_identity_digest_anchor",
    )
    monkeypatch.setattr(
        run_embed_module.infer_runtime,
        "run_sd3_inference",
        lambda *args, **kwargs: {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {},
            "trajectory_evidence": {},
            "injection_evidence": {},
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        },
    )

    with pytest.raises(_StopAfterContentExtract):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    assert isinstance(captured_inputs.get("inputs"), dict)
    preview_path = captured_inputs["inputs"].get("image_path")
    assert isinstance(preview_path, str)
    assert preview_path.endswith(".png")
    assert Path(preview_path).exists()


def test_run_embed_runtime_finalization_planner_inputs_include_precontent_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：runtime finalization planner 必须消费 pre-content payload 中的 routing_digest 与 mask_summary。
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    @contextmanager
    def _bound_fact_sources(*args, **kwargs):
        _ = args
        _ = kwargs
        yield

    def _ensure_output_layout(path: Path, **kwargs):
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

    cfg_payload = {
        "policy_path": "content_np_geo_rescue",
        "paper_faithfulness": {"enabled": True},
        "attestation": {"enabled": True, "use_trajectory_mix": False},
        "watermark": {
            "hf": {"enabled": False},
            "lf": {"enabled": True, "strength": 0.5},
            "subspace": {"enabled": True, "rank": 8, "sample_count": 4, "feature_dim": 16},
        },
        "inference_num_steps": 4,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
    }

    captured_planner_inputs = {}

    class _FakeContentExtractor:
        impl_version = "v1"

        def extract(self, cfg, inputs=None, cfg_digest=None):
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
        def plan(self, cfg, mask_digest=None, cfg_digest=None, inputs=None):
            _ = cfg
            _ = cfg_digest
            captured_planner_inputs["mask_digest"] = mask_digest
            captured_planner_inputs["inputs"] = inputs
            return _FakeSubspacePlanResult(
                status="ok",
                plan={
                    "basis_digest": "basis_digest_anchor",
                    "lf_basis": {"basis_digest": "basis_digest_anchor"},
                    "planner_params": {"rank": 8},
                },
                plan_digest="plan_digest_anchor",
                basis_digest="basis_digest_anchor",
            )

    class _FakeImplSet:
        def __init__(self):
            self.content_extractor = _FakeContentExtractor()
            self.subspace_planner = _FakeSubspacePlanner()

    class _FakeImplIdentity:
        content_extractor_id = "unified_content_extractor"

        def as_dict(self):
            return {"content_extractor_id": self.content_extractor_id}

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", lambda _: run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", lambda *_: {"contracts": "ok"})
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", lambda *_: {"whitelist": "ok"})
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", lambda *_: {"semantics": "ok"})
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", lambda: {"manifest": "ok"})
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", lambda *args, **kwargs: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", lambda: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", lambda *_: SimpleNamespace())
    monkeypatch.setattr(
        run_embed_module.config_loader,
        "load_and_validate_config",
        lambda *args, **kwargs: (dict(cfg_payload), "cfg_digest_anchor", {"cfg_pruned_for_digest_canon_sha256": "a", "cfg_audit_canon_sha256": "b"}),
    )
    monkeypatch.setattr(run_embed_module, "build_seed_audit", lambda *args, **kwargs: ({}, "seed_digest", 7, "seed_rule"))
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", lambda *_: None)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", lambda *_: None)
    monkeypatch.setattr(
        run_embed_module.pipeline_factory,
        "build_pipeline_shell",
        lambda *_: {
            "pipeline_obj": object(),
            "pipeline_provenance_canon_sha256": "<absent>",
            "pipeline_status": "built",
            "pipeline_error": "<absent>",
            "pipeline_runtime_meta": {},
            "env_fingerprint_canon_sha256": "<absent>",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "<absent>",
        },
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "build_runtime_impl_set_from_cfg",
        lambda *_: (_FakeImplIdentity(), _FakeImplSet(), "impl_cap_digest"),
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "compute_impl_identity_digest",
        lambda *_: "impl_identity_digest_anchor",
    )
    def _fake_runtime_capture(*args, **kwargs):
        trajectory_cache = kwargs.get("trajectory_latent_cache")
        if trajectory_cache is not None:
            trajectory_cache.capture(0, np.ones((1, 4, 4, 4), dtype=np.float32))
            trajectory_cache.capture(1, np.full((1, 4, 4, 4), 2.0, dtype=np.float32))
            trajectory_cache.capture(2, np.full((1, 4, 4, 4), 3.0, dtype=np.float32))
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {},
            "trajectory_evidence": {},
            "injection_evidence": {},
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    monkeypatch.setattr(run_embed_module.infer_runtime, "run_sd3_inference", _fake_runtime_capture)
    monkeypatch.setattr(
        run_embed_module.embed_orchestrator,
        "_prepare_embed_attestation_runtime_bindings",
        lambda *args, **kwargs: (_ for _ in ()).throw(_StopAfterPlannerPrecompute("stop after planner precompute")),
    )

    with pytest.raises(_StopAfterPlannerPrecompute):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    assert captured_planner_inputs["mask_digest"] == "mask_digest_anchor"
    assert captured_planner_inputs["inputs"]["routing_digest"] == "routing_digest_anchor"
    assert captured_planner_inputs["inputs"]["mask_summary"]["routing_digest"] == "routing_digest_anchor"
    assert callable(captured_planner_inputs["inputs"]["jvp_operator"])


def test_run_embed_statement_only_formal_path_fails_fast_when_runtime_finalization_plan_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：statement_only formal path 若 runtime finalization 拿不到 executable plan，必须 fail-fast，不能进入 fallback plan。
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    @contextmanager
    def _bound_fact_sources(*args, **kwargs):
        _ = args
        _ = kwargs
        yield

    def _ensure_output_layout(path: Path, **kwargs):
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

    cfg_payload = {
        "policy_path": "content_np_geo_rescue",
        "paper_faithfulness": {"enabled": True},
        "attestation": {"enabled": True, "use_trajectory_mix": False},
        "watermark": {
            "hf": {"enabled": False},
            "lf": {"enabled": True, "strength": 0.5},
            "subspace": {"enabled": True, "rank": 8, "sample_count": 4, "feature_dim": 16},
        },
        "inference_num_steps": 4,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
    }
    runtime_capture_calls = {"count": 0}

    class _FakeContentExtractor:
        impl_version = "v1"

        def extract(self, cfg, inputs=None, cfg_digest=None):
            _ = cfg
            _ = inputs
            _ = cfg_digest
            return {
                "status": "ok",
                "mask_digest": "mask_digest_anchor",
                "mask_stats": {
                    "area_ratio": 0.25,
                    "routing_digest": "routing_digest_anchor",
                },
            }

    class _FakeSubspacePlanner:
        def plan(self, cfg, mask_digest=None, cfg_digest=None, inputs=None):
            _ = cfg
            _ = mask_digest
            _ = cfg_digest
            runtime_capture_calls["planner_inputs"] = inputs
            return _FakeSubspacePlanResult(
                status="failed",
                plan=None,
                plan_digest=None,
                basis_digest=None,
                plan_failure_reason="runtime_jvp_operator_required",
            )

    class _FakeImplSet:
        def __init__(self):
            self.content_extractor = _FakeContentExtractor()
            self.subspace_planner = _FakeSubspacePlanner()

    class _FakeImplIdentity:
        content_extractor_id = "unified_content_extractor"

        def as_dict(self):
            return {"content_extractor_id": self.content_extractor_id}

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", lambda _: run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", lambda *_: {"contracts": "ok"})
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", lambda *_: {"whitelist": "ok"})
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", lambda *_: {"semantics": "ok"})
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", lambda: {"manifest": "ok"})
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", lambda *args, **kwargs: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", lambda: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", lambda *_: SimpleNamespace())
    monkeypatch.setattr(
        run_embed_module.config_loader,
        "load_and_validate_config",
        lambda *args, **kwargs: (dict(cfg_payload), "cfg_digest_anchor", {"cfg_pruned_for_digest_canon_sha256": "a", "cfg_audit_canon_sha256": "b"}),
    )
    monkeypatch.setattr(run_embed_module, "build_seed_audit", lambda *args, **kwargs: ({}, "seed_digest", 7, "seed_rule"))
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", lambda *_: None)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", lambda *_: None)
    monkeypatch.setattr(
        run_embed_module.pipeline_factory,
        "build_pipeline_shell",
        lambda *_: {
            "pipeline_obj": object(),
            "pipeline_provenance_canon_sha256": "<absent>",
            "pipeline_status": "built",
            "pipeline_error": "<absent>",
            "pipeline_runtime_meta": {},
            "env_fingerprint_canon_sha256": "<absent>",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "<absent>",
        },
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "build_runtime_impl_set_from_cfg",
        lambda *_: (_FakeImplIdentity(), _FakeImplSet(), "impl_cap_digest"),
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "compute_impl_identity_digest",
        lambda *_: "impl_identity_digest_anchor",
    )
    def _fake_runtime_capture(*args, **kwargs):
        runtime_capture_calls["count"] += 1
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {},
            "trajectory_evidence": {},
            "injection_evidence": {},
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        }

    monkeypatch.setattr(
        run_embed_module.infer_runtime,
        "run_sd3_inference",
        _fake_runtime_capture,
    )

    with pytest.raises(ValueError, match="runtime executable formal plan unavailable: statement_only runtime finalization failed; reason=runtime_jvp_operator_required"):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    assert runtime_capture_calls["count"] == 1
    assert "jvp_operator" not in runtime_capture_calls["planner_inputs"]


def test_run_embed_statement_only_formal_path_reaches_orchestrator_with_runtime_finalized_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：statement_only 两阶段 formal object 成功时，workflow 必须越过 embed gate 并进入 orchestrator。
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"

    @contextmanager
    def _bound_fact_sources(*args, **kwargs):
        _ = args
        _ = kwargs
        yield

    def _ensure_output_layout(path: Path, **kwargs):
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

    cfg_payload = {
        "policy_path": "content_np_geo_rescue",
        "paper_faithfulness": {"enabled": True},
        "attestation": {"enabled": True, "use_trajectory_mix": False},
        "watermark": {
            "hf": {"enabled": False},
            "lf": {"enabled": True, "strength": 0.5},
            "subspace": {"enabled": True, "rank": 8, "sample_count": 4, "feature_dim": 16},
        },
        "inference_num_steps": 4,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
    }

    captured = {"inference_calls": 0}

    class _FakeContentExtractor:
        impl_version = "v1"

        def extract(self, cfg, inputs=None, cfg_digest=None):
            _ = cfg
            _ = inputs
            _ = cfg_digest
            return {
                "status": "ok",
                "mask_digest": "mask_digest_anchor",
                "mask_stats": {
                    "area_ratio": 0.25,
                    "routing_digest": "routing_digest_anchor",
                },
            }

    class _FakeSubspacePlanner:
        def plan(self, cfg, mask_digest=None, cfg_digest=None, inputs=None):
            _ = cfg
            _ = mask_digest
            _ = cfg_digest
            captured["planner_inputs"] = inputs
            return _FakeSubspacePlanResult(
                status="ok",
                plan={
                    "basis_digest": "final_basis_digest_anchor",
                    "lf_basis": {"basis_digest": "final_basis_digest_anchor"},
                    "planner_params": {"rank": 8},
                },
                plan_digest="final_plan_digest_anchor",
                basis_digest="final_basis_digest_anchor",
            )

    class _FakeImplSet:
        def __init__(self):
            self.content_extractor = _FakeContentExtractor()
            self.subspace_planner = _FakeSubspacePlanner()

    class _FakeImplIdentity:
        content_extractor_id = "unified_content_extractor"

        def as_dict(self):
            return {"content_extractor_id": self.content_extractor_id}

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", lambda _: run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", lambda *_: {"contracts": "ok"})
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", lambda *_: {"whitelist": "ok"})
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", lambda *_: {"semantics": "ok"})
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", lambda: {"manifest": "ok"})
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", lambda *args, **kwargs: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", lambda: {"snapshot": "ok"})
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", lambda *_: SimpleNamespace())
    monkeypatch.setattr(
        run_embed_module.config_loader,
        "load_and_validate_config",
        lambda *args, **kwargs: (dict(cfg_payload), "cfg_digest_anchor", {"cfg_pruned_for_digest_canon_sha256": "a", "cfg_audit_canon_sha256": "b"}),
    )
    monkeypatch.setattr(run_embed_module, "build_seed_audit", lambda *args, **kwargs: ({}, "seed_digest", 7, "seed_rule"))
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", lambda *_: None)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", lambda *_: None)
    monkeypatch.setattr(
        run_embed_module.pipeline_factory,
        "build_pipeline_shell",
        lambda *_: {
            "pipeline_obj": object(),
            "pipeline_provenance_canon_sha256": "<absent>",
            "pipeline_status": "built",
            "pipeline_error": "<absent>",
            "pipeline_runtime_meta": {},
            "env_fingerprint_canon_sha256": "<absent>",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "<absent>",
        },
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "build_runtime_impl_set_from_cfg",
        lambda *_: (_FakeImplIdentity(), _FakeImplSet(), "impl_cap_digest"),
    )
    monkeypatch.setattr(
        run_embed_module.runtime_resolver,
        "compute_impl_identity_digest",
        lambda *_: "impl_identity_digest_anchor",
    )

    def _fake_inference(*args, **kwargs):
        captured["inference_calls"] += 1
        trajectory_cache = kwargs.get("trajectory_latent_cache")
        injection_context = kwargs.get("injection_context")
        if captured["inference_calls"] == 1:
            assert injection_context is None
            if trajectory_cache is not None:
                trajectory_cache.capture(0, np.ones((1, 4, 4, 4), dtype=np.float32))
                trajectory_cache.capture(1, np.full((1, 4, 4, 4), 2.0, dtype=np.float32))
                trajectory_cache.capture(2, np.full((1, 4, 4, 4), 3.0, dtype=np.float32))
        else:
            assert injection_context is not None
            captured["injection_plan_digest"] = injection_context.plan_digest
        return {
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {},
            "trajectory_evidence": {},
            "injection_evidence": {"status": "ok"},
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
            "final_latents": None,
        }

    monkeypatch.setattr(run_embed_module.infer_runtime, "run_sd3_inference", _fake_inference)

    def _fake_run_embed_orchestrator(cfg, impl_set, cfg_digest, **kwargs):
        _ = impl_set
        _ = cfg_digest
        captured["formal_scaffold_digest"] = cfg.get("__formal_scaffold__", {}).get("formal_scaffold_digest")
        override = kwargs.get("subspace_result_override")
        captured["override_plan_digest"] = getattr(override, "plan_digest", None)
        raise _StopAfterRunEmbedOrchestrator("entered orchestrator")

    monkeypatch.setattr(run_embed_module, "run_embed_orchestrator", _fake_run_embed_orchestrator)

    with pytest.raises(_StopAfterRunEmbedOrchestrator):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    assert captured["inference_calls"] == 2
    assert callable(captured["planner_inputs"]["jvp_operator"])
    assert captured["injection_plan_digest"] == "final_plan_digest_anchor"
    assert captured["override_plan_digest"] == "final_plan_digest_anchor"
    assert captured["formal_scaffold_digest"] != "final_plan_digest_anchor"
