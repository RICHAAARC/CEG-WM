"""
文件目的：验证 run_embed 的 preview_generation 正式工件契约与失败可观测性。
Module type: General module
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterator, cast

import numpy as np
from PIL import Image
import pytest

from main.cli import run_embed as run_embed_module
from main.diffusion.sd3 import diffusers_loader, pipeline_factory


build_statement_only_formal_scaffold = run_embed_module._build_statement_only_formal_scaffold  # pyright: ignore[reportPrivateUsage]


class StopAfterContentExtract(Exception):
    """Sentinel exception used to stop run_embed after content precompute."""


class FakeContentExtractor:
    """Minimal content extractor that captures precompute inputs."""

    impl_version = "v1"

    def __init__(self, captured_inputs: Dict[str, Any]) -> None:
        self._captured_inputs = captured_inputs

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Dict[str, Any] | None = None,
        cfg_digest: str | None = None,
    ) -> Dict[str, Any]:
        """
        功能：捕获 content precompute 输入并中止执行。

        Capture the content precompute inputs and stop the run.

        Args:
            cfg: Runtime configuration.
            inputs: Content extractor inputs.
            cfg_digest: Config digest.

        Returns:
            None.
        """
        _ = cfg
        _ = cfg_digest
        self._captured_inputs["inputs"] = inputs
        raise StopAfterContentExtract("stop after preview content precompute")


class FakeImplSet:
    """Minimal implementation set required by run_embed preview tests."""

    def __init__(self, captured_inputs: Dict[str, Any]) -> None:
        self.content_extractor = FakeContentExtractor(captured_inputs)


class FakeImplIdentity:
    """Minimal impl identity payload required by run_embed preview tests."""

    content_extractor_id = "unified_content_extractor_v1"

    def as_dict(self) -> Dict[str, str]:
        """
        功能：返回最小 impl identity 字典。

        Return the minimal impl identity mapping.

        Args:
            None.

        Returns:
            Minimal identity mapping.
        """
        return {"content_extractor_id": self.content_extractor_id}


def _prepare_run_embed_preview_monkeypatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cfg_payload: Dict[str, Any],
    captured_inputs: Dict[str, Any],
    inference_result: Dict[str, Any] | None = None,
    inference_exception: Exception | None = None,
) -> Dict[str, Any]:
    """
    功能：为 preview_generation 回归测试配置 run_embed 最小 monkeypatch 环境。

    Prepare the minimal monkeypatch environment for run_embed preview-generation
    regression tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.
        cfg_payload: Runtime config payload.
        captured_inputs: Mutable capture dictionary.
        inference_result: Structured fake inference result.
        inference_exception: Optional inference exception.

    Returns:
        Runtime path and call-order captures.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    logs_dir = run_root / "logs"
    call_order: list[str] = []

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
        call_order.append("layout")
        return {
            "run_root": path,
            "records_dir": records_dir,
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
        }

    def _fake_run_sd3_inference(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = args
        _ = kwargs
        call_order.append("preview_inference")
        if inference_exception is not None:
            raise inference_exception
        if inference_result is None:
            raise RuntimeError("inference_result must be provided when inference_exception is None")
        return inference_result

    def _derive_run_root(_output_dir: Any) -> Path:
        return run_root

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
            dict(cfg_payload),
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
            "pipeline_provenance_canon_sha256": "<absent>",
            "pipeline_status": "built",
            "pipeline_error": None,
            "pipeline_runtime_meta": {
                "model_source_resolution": "fallback_to_requested_model_source",
                "local_snapshot_status": "absent",
            },
            "env_fingerprint_canon_sha256": "<absent>",
            "diffusers_version": "<absent>",
            "transformers_version": "<absent>",
            "safetensors_version": "<absent>",
            "model_provenance_canon_sha256": "<absent>",
        }

    def _build_runtime_impl_set_from_cfg(*args: Any) -> tuple[FakeImplIdentity, FakeImplSet, str]:
        _ = args
        return (FakeImplIdentity(), FakeImplSet(captured_inputs), "impl_cap_digest_anchor")

    def _compute_impl_identity_digest(*args: Any) -> str:
        _ = args
        return "impl_identity_digest_anchor"

    monkeypatch.setattr(run_embed_module.path_policy, "derive_run_root", _derive_run_root)
    monkeypatch.setattr(run_embed_module.path_policy, "ensure_output_layout", _ensure_output_layout)
    monkeypatch.setattr(run_embed_module.path_policy, "anchor_requirements", _anchor_requirements)
    monkeypatch.setattr(run_embed_module.status, "finalize_run", _finalize_run)
    monkeypatch.setattr(run_embed_module, "load_frozen_contracts", _load_frozen_contracts)
    monkeypatch.setattr(run_embed_module, "load_runtime_whitelist", _load_runtime_whitelist)
    monkeypatch.setattr(run_embed_module, "load_policy_path_semantics", _load_policy_path_semantics)
    monkeypatch.setattr(run_embed_module.config_loader, "load_injection_scope_manifest", cast(Any, lambda: {"manifest": "ok"}))
    monkeypatch.setattr(run_embed_module.status, "bind_freeze_anchors_to_run_meta", _bind_freeze_anchors_to_run_meta)
    monkeypatch.setattr(run_embed_module.records_io, "build_fact_sources_snapshot", _build_fact_sources_snapshot)
    monkeypatch.setattr(run_embed_module, "assert_consistent_with_semantics", _assert_consistent_with_semantics)
    monkeypatch.setattr(run_embed_module.records_io, "bound_fact_sources", _bound_fact_sources)
    monkeypatch.setattr(run_embed_module.records_io, "get_bound_fact_sources", cast(Any, lambda: {"snapshot": "ok"}))
    monkeypatch.setattr(run_embed_module, "get_contract_interpretation", _get_contract_interpretation)
    monkeypatch.setattr(run_embed_module.config_loader, "load_and_validate_config", _load_and_validate_config)
    monkeypatch.setattr(run_embed_module, "build_seed_audit", _build_seed_audit)
    monkeypatch.setattr(run_embed_module, "build_determinism_controls", _build_determinism_controls)
    monkeypatch.setattr(run_embed_module, "normalize_nondeterminism_notes", _normalize_nondeterminism_notes)
    monkeypatch.setattr(run_embed_module.pipeline_factory, "build_pipeline_shell", _build_pipeline_shell)
    monkeypatch.setattr(run_embed_module.runtime_resolver, "build_runtime_impl_set_from_cfg", _build_runtime_impl_set_from_cfg)
    monkeypatch.setattr(run_embed_module.runtime_resolver, "compute_impl_identity_digest", _compute_impl_identity_digest)
    monkeypatch.setattr(run_embed_module.infer_runtime, "run_sd3_inference", _fake_run_sd3_inference)

    return {
        "run_root": run_root,
        "records_dir": records_dir,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
        "call_order": call_order,
    }


def test_run_embed_preview_generation_persists_formal_artifact_before_content_precompute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：preview_generation 必须先进入正式 output 布局，再把预览图落到受控 artifact 路径。
    """
    captured_inputs: Dict[str, Any] = {}
    cfg_payload: Dict[str, Any] = {
        "policy_path": "content_only",
        "embed": {
            "preview_generation": {
                "enabled": True,
                "artifact_rel_path": "preview/preview.png",
            },
        },
        "detect": {"content": {"enabled": True}},
        "watermark": {"hf": {"enabled": False}},
        "inference_enabled": True,
        "inference_prompt": "preview prompt",
        "inference_num_steps": 1,
        "inference_guidance_scale": 1.0,
        "inference_height": 8,
        "inference_width": 8,
        "device": "cpu",
    }
    env = _prepare_run_embed_preview_monkeypatches(
        monkeypatch,
        tmp_path,
        cfg_payload=cfg_payload,
        captured_inputs=captured_inputs,
        inference_result={
            "inference_status": "ok",
            "inference_error": None,
            "inference_runtime_meta": {
                "latency_ms": 1.0,
                "cuda_memory_profile": {
                    "status": "absent",
                    "reason": "cuda_not_active",
                    "phase_label": "preview_generation",
                    "sample_scope": "single_worker_process_local",
                    "device": "cpu",
                },
                "nested_runtime": {
                    "preview_token": "kept",
                    "preview_tensor": np.ones((1, 4, 8, 8), dtype=np.float32),
                },
                "preview_tensor": np.ones((1, 4, 8, 8), dtype=np.float32),
            },
            "trajectory_evidence": {},
            "injection_evidence": {},
            "output_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
        },
    )

    with pytest.raises(StopAfterContentExtract):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    preview_artifact_path = env["artifacts_dir"] / "preview" / "preview.png"
    preview_record_path = env["artifacts_dir"] / "preview" / "preview_generation_record.json"
    preview_record = json.loads(preview_record_path.read_text(encoding="utf-8"))

    assert env["call_order"][:2] == ["layout", "preview_inference"]
    assert captured_inputs["inputs"]["image_path"] == str(preview_artifact_path)
    assert captured_inputs["inputs"]["content_runtime_phase"] == run_embed_module.EMBED_CONTENT_RUNTIME_PHASE_PRECOMPUTE
    assert preview_artifact_path.exists()
    assert preview_record["status"] == "ok"
    assert preview_record["requested_artifact_rel_path"] == "preview/preview.png"
    assert preview_record["persisted_artifact_rel_path"] == "preview/preview.png"
    assert preview_record["persisted_artifact_path"] == str(preview_artifact_path)
    assert preview_record["record_rel_path"] == "preview/preview_generation_record.json"
    assert preview_record["output_image_present"] is True
    assert preview_record["inference_runtime_meta"] == {
        "latency_ms": 1.0,
        "cuda_memory_profile": {
            "status": "absent",
            "reason": "cuda_not_active",
            "phase_label": "preview_generation",
            "sample_scope": "single_worker_process_local",
            "device": "cpu",
        },
        "nested_runtime": {"preview_token": "kept"},
    }
    assert preview_record["inference_runtime_meta"]["cuda_memory_profile"]["status"] == "absent"
    assert preview_record["pipeline_runtime_meta"]["model_source_resolution"] == "fallback_to_requested_model_source"
    assert preview_record["pipeline_runtime_meta"]["local_snapshot_status"] == "absent"
    assert preview_record["seed"] == 7


def test_run_embed_preview_generation_failure_writes_structured_observability_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：preview_generation 失败时必须写出结构化 record，并保持 formal scaffold 输入不被伪造。
    """
    captured_inputs: Dict[str, Any] = {}
    cfg_payload: Dict[str, Any] = {
        "policy_path": "content_only",
        "embed": {
            "preview_generation": {
                "enabled": True,
                "artifact_rel_path": "preview/preview.png",
            },
        },
        "detect": {"content": {"enabled": True}},
        "watermark": {"hf": {"enabled": False}},
        "inference_enabled": True,
        "inference_prompt": "preview prompt",
        "inference_num_steps": 1,
        "inference_guidance_scale": 1.0,
        "inference_height": 8,
        "inference_width": 8,
        "device": "cpu",
    }
    env = _prepare_run_embed_preview_monkeypatches(
        monkeypatch,
        tmp_path,
        cfg_payload=cfg_payload,
        captured_inputs=captured_inputs,
        inference_exception=RuntimeError("preview backend offline"),
    )

    with pytest.raises(StopAfterContentExtract):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/default.yaml",
            overrides=None,
            input_image_path=None,
        )

    preview_record_path = env["artifacts_dir"] / "preview" / "preview_generation_record.json"
    preview_record = json.loads(preview_record_path.read_text(encoding="utf-8"))

    assert env["call_order"][:2] == ["layout", "preview_inference"]
    assert captured_inputs["inputs"].get("image_path") is None
    assert preview_record["status"] == "failed"
    assert preview_record["reason"] == "preview_generation_exception"
    assert preview_record["exception_type"] == "RuntimeError"
    assert preview_record["exception_message"] == "preview backend offline"
    assert preview_record["persisted_artifact_path"] is None
    assert preview_record["persisted_artifact_rel_path"] is None
    assert preview_record["output_image_present"] is False
    assert preview_record["record_rel_path"] == "preview/preview_generation_record.json"
    assert preview_record["pipeline_runtime_meta"]["model_source_resolution"] == "fallback_to_requested_model_source"


def test_build_statement_only_formal_scaffold_requires_mask_digest() -> None:
    """
    功能：statement_only formal scaffold 缺少 mask_digest 时必须 fail-fast，不能被 preview 路径放宽。
    """
    scaffold, failure_reason = build_statement_only_formal_scaffold(
        {"policy_path": "content_np_geo_rescue", "model_id": "sd3-test", "inference_prompt": "prompt"},
        "cfg_digest_anchor",
        7,
        {
            "status": "failed",
            "mask_stats": {"routing_digest": "routing_digest_anchor", "area_ratio": 0.25},
        },
        {
            "trace_signature": {"num_inference_steps": 4},
            "mask_summary": {"routing_digest": "routing_digest_anchor", "area_ratio": 0.25},
            "routing_digest": "routing_digest_anchor",
        },
    )

    assert scaffold is None
    assert failure_reason == "scaffold_mask_digest_absent"


def test_build_sd3_pipeline_from_pretrained_prefers_bound_local_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能：验证 loader 在 notebook 绑定的本地 snapshot 有效时优先使用本地目录。

    Verify the SD3 loader prefers the bound local snapshot directory when the
    notebook-provided path is valid.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    captured: Dict[str, Any] = {}

    class FakePipeline:
        def to(self, device: str) -> "FakePipeline":
            captured["device"] = device
            return self

    class FakeDiffusionPipeline:
        @staticmethod
        def from_pretrained(model_ref: str, **build_kwargs: Any) -> FakePipeline:
            captured["model_ref"] = model_ref
            captured["build_kwargs"] = build_kwargs
            return FakePipeline()

    fake_diffusers = SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline)
    monkeypatch.setattr(
        diffusers_loader,
        "try_import_diffusers",
        lambda: (
            True,
            {
                "diffusers_version": "0.test",
                "transformers_version": "0.test",
                "safetensors_version": "0.test",
            },
        ),
    )
    monkeypatch.setattr(diffusers_loader.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipeline_obj, build_meta, error = diffusers_loader.build_sd3_pipeline_from_pretrained(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        revision="main",
        model_source="hf",
        extra_kwargs={"device": "cpu"},
        local_snapshot_path=snapshot_dir.as_posix(),
    )

    assert error is None
    assert pipeline_obj is not None
    assert captured["model_ref"] == snapshot_dir.as_posix()
    assert captured["build_kwargs"]["local_files_only"] is True
    assert build_meta["resolved_model_id"] == snapshot_dir.as_posix()
    assert build_meta["resolved_model_source"] == "local_path"
    assert build_meta["model_source_resolution"] == "local_snapshot_priority"
    assert build_meta["local_snapshot_status"] == "bound"
    assert build_meta["requested_model_source"] == "hf"


def test_build_sd3_pipeline_from_pretrained_records_invalid_local_snapshot_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 loader 在本地 snapshot 无效时回退到请求 source，并保留可观测元数据。

    Verify the SD3 loader falls back to the requested source when the bound
    local snapshot path is invalid and records the fallback metadata.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    invalid_snapshot_dir = tmp_path / "missing_snapshot"
    captured: Dict[str, Any] = {}

    class FakePipeline:
        def to(self, device: str) -> "FakePipeline":
            captured["device"] = device
            return self

    class FakeDiffusionPipeline:
        @staticmethod
        def from_pretrained(model_ref: str, **build_kwargs: Any) -> FakePipeline:
            captured["model_ref"] = model_ref
            captured["build_kwargs"] = build_kwargs
            return FakePipeline()

    fake_diffusers = SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline)
    monkeypatch.setattr(
        diffusers_loader,
        "try_import_diffusers",
        lambda: (
            True,
            {
                "diffusers_version": "0.test",
                "transformers_version": "0.test",
                "safetensors_version": "0.test",
            },
        ),
    )
    monkeypatch.setattr(diffusers_loader.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipeline_obj, build_meta, error = diffusers_loader.build_sd3_pipeline_from_pretrained(
        model_id="stabilityai/stable-diffusion-3.5-medium",
        revision="main",
        model_source="hf",
        extra_kwargs={"device": "cpu"},
        local_snapshot_path=invalid_snapshot_dir.as_posix(),
    )

    assert error is None
    assert pipeline_obj is not None
    assert captured["model_ref"] == "stabilityai/stable-diffusion-3.5-medium"
    assert captured["build_kwargs"]["local_files_only"] is False
    assert build_meta["resolved_model_id"] == "stabilityai/stable-diffusion-3.5-medium"
    assert build_meta["resolved_model_source"] == "hf"
    assert build_meta["model_source_resolution"] == "fallback_to_requested_model_source"
    assert build_meta["local_snapshot_status"] == "invalid"
    assert build_meta["local_snapshot_error"] == "local_snapshot_path_missing_or_not_directory"


def test_build_pipeline_shell_exposes_local_snapshot_resolution_in_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 pipeline factory 会把 local snapshot 解析结果写入 runtime meta 与 provenance。

    Verify the pipeline factory threads the local snapshot resolution into both
    runtime metadata and pipeline provenance.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary pytest directory.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    captured: Dict[str, Any] = {}

    monkeypatch.setattr(
        pipeline_factory,
        "preflight_pipeline_build",
        lambda _cfg: (True, None, None),
    )
    monkeypatch.setattr(
        pipeline_factory.diffusers_loader,
        "try_import_diffusers",
        lambda: (
            True,
            {
                "diffusers_version": "0.test",
                "transformers_version": "0.test",
                "safetensors_version": "0.test",
            },
        ),
    )

    def _fake_build_sd3_pipeline_from_pretrained(**kwargs: Any) -> tuple[Any, Dict[str, Any], str | None]:
        captured.update(kwargs)
        return object(), {
            "status": "built",
            "local_files_only": True,
            "resolved_model_id": snapshot_dir.as_posix(),
            "resolved_model_source": "local_path",
            "model_source_resolution": "local_snapshot_priority",
            "local_snapshot_path": snapshot_dir.as_posix(),
            "local_snapshot_status": "bound",
            "local_snapshot_error": "<absent>",
        }, None

    monkeypatch.setattr(
        pipeline_factory.diffusers_loader,
        "build_sd3_pipeline_from_pretrained",
        _fake_build_sd3_pipeline_from_pretrained,
    )
    monkeypatch.setattr(
        pipeline_factory.weights_snapshot,
        "compute_weights_snapshot_sha256",
        lambda **_kwargs: ("weights_digest", {"snapshot_status": "built"}, None),
    )
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "build_env_fingerprint",
        lambda: {"env": "ok"},
    )
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "compute_env_fingerprint_canon_sha256",
        lambda _env: "env_digest",
    )

    result = pipeline_factory.build_pipeline_shell(
        {
            "pipeline_impl_id": pipeline_factory.pipeline_registry.SD3_DIFFUSERS_REAL_ID,
            "paper_faithfulness": {"enabled": True},
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "model_source": "hf",
            "hf_revision": "main",
            "model_snapshot_path": snapshot_dir.as_posix(),
            "device": "cpu",
            "model": {"dtype": "float32"},
            "model_source_binding": {
                "binding_status": "bound",
                "binding_reason": "model_snapshot_env_var_bound_to_runtime_config",
            },
        }
    )

    assert captured["local_snapshot_path"] == snapshot_dir.as_posix()
    assert result["pipeline_status"] == "built"
    assert result["pipeline_runtime_meta"]["model_source_resolution"] == "local_snapshot_priority"
    assert result["pipeline_runtime_meta"]["local_snapshot_status"] == "bound"
    assert result["pipeline_runtime_meta"]["model_source_binding"]["binding_status"] == "bound"
    assert result["pipeline_provenance"]["model_source"] == "hf"
    assert result["pipeline_provenance"]["resolved_model_source"] == "local_path"
    assert result["pipeline_provenance"]["resolved_model_id"] == snapshot_dir.as_posix()
    assert result["pipeline_provenance"]["model_source_resolution"] == "local_snapshot_priority"
    assert result["pipeline_provenance"]["local_snapshot_status"] == "bound"
