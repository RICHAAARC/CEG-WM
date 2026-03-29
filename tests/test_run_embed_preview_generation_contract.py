"""
文件目的：验证 run_embed 的 preview_generation 正式工件契约与失败可观测性。
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
            "pipeline_runtime_meta": {},
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
            "inference_runtime_meta": {"latency_ms": 1.0},
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
