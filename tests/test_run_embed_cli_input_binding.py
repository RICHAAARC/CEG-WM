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
