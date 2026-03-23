"""
File purpose: run_embed real pipeline 身份字段前置校验回归测试。
Module type: General module
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from main.cli import run_embed as run_embed_module


def test_run_embed_fails_before_pipeline_shell_when_model_source_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：缺失 model_source 时必须在 run_embed 入口前置失败。

    Verify run_embed raises a clear config error before pipeline shell
    construction when the real SD3 pipeline lacks model_source.

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
        "policy_path": "content_np_geo_rescue",
        "pipeline_impl_id": "sd3_diffusers_real",
        "pipeline_build_enabled": True,
        "paper_faithfulness": {"enabled": True},
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "hf_revision": "main",
    }

    pipeline_factory_called = {"value": False}

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

    def _unexpected_pipeline_shell(*args, **kwargs):
        _ = args
        _ = kwargs
        pipeline_factory_called["value"] = True
        raise AssertionError("pipeline_factory.build_pipeline_shell must not be reached")

    monkeypatch.setattr(
        run_embed_module.pipeline_factory,
        "build_pipeline_shell",
        _unexpected_pipeline_shell,
    )

    with pytest.raises(ValueError, match="model_source"):
        run_embed_module.run_embed(
            output_dir=str(tmp_path / "out"),
            config_path="configs/paper_full_cuda.yaml",
            overrides=None,
            input_image_path=None,
        )

    assert pipeline_factory_called["value"] is False