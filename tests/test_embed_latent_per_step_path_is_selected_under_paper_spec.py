"""
文件目的：paper 配置下 embed 必走 latent per-step 路径回归测试。
Module type: General module
"""

import pytest

from main.watermarking.embed.orchestrator import (
    _build_latent_step_embed_trace,
    _should_use_latent_per_step_path,
)


def test_embed_latent_per_step_path_is_selected_under_paper_spec() -> None:
    cfg = {
        "paper_faithfulness": {
            "enabled": True,
        }
    }
    assert _should_use_latent_per_step_path(cfg) is True

    trace = _build_latent_step_embed_trace(
        cfg,
        {
            "status": "ok",
            "injection_trace_digest": "a" * 64,
            "injection_params_digest": "b" * 64,
        },
    )
    assert trace["embed_mode"] == "latent_step_injection_v1"
    assert trace["injection_status"] == "ok"


def test_embed_latent_per_step_path_requires_ok_injection_evidence() -> None:
    cfg = {
        "paper_faithfulness": {
            "enabled": True,
        }
    }
    with pytest.raises(ValueError):
        _build_latent_step_embed_trace(cfg, {"status": "absent"})
