"""
File purpose: 验证 LatentSyncGeometryExtractor 不会绕过 records_io 写盘。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from main.core import records_io
from main.watermarking.geometry_chain.sync.latent_sync_template import (
    GEOMETRY_LATENT_SYNC_SD3_ID,
    GEOMETRY_LATENT_SYNC_SD3_VERSION,
    LatentSyncGeometryExtractor,
)


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def test_latent_sync_extractor_must_not_write_records_or_artifacts(monkeypatch: MonkeyPatch) -> None:
    """
    功能：同步模板提取器不得直接调用 records_io 写盘入口。

    Sync extractor must not directly call records_io write APIs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    called = {"count": 0}

    def _forbidden_call(*args: Any, **kwargs: Any) -> None:
        called["count"] += 1
        raise AssertionError("records_io write API must not be called by sync extractor")

    monkeypatch.setattr(records_io, "write_json", _forbidden_call)
    monkeypatch.setattr(records_io, "write_artifact_json", _forbidden_call)
    monkeypatch.setattr(records_io, "copy_file_controlled", _forbidden_call)

    extractor = LatentSyncGeometryExtractor(GEOMETRY_LATENT_SYNC_SD3_ID, GEOMETRY_LATENT_SYNC_SD3_VERSION, "a" * 64)
    cfg: Dict[str, Any] = {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_latent_sync": True,
            }
        },
    }

    latents = np.random.RandomState(10).randn(1, 4, 8, 8).astype(np.float32)
    evidence = extractor.extract(cfg, inputs={"pipeline": _Pipeline(), "latents": latents})

    assert evidence.get("status") == "ok"
    assert called["count"] == 0
