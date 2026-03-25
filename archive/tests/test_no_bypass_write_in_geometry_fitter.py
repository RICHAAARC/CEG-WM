"""
File purpose: 验证几何拟合器路径不会绕过 records_io 写盘。
Module type: General module
"""

from __future__ import annotations

from typing import Any

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from main.core import records_io
from main.watermarking.geometry_chain.align_invariance_extractor import GeometryAlignInvarianceExtractor


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def test_no_bypass_write_in_geometry_fitter(monkeypatch: MonkeyPatch) -> None:
    """
    功能：稳健拟合升级后，几何链仍不得直接调用 records_io 写盘入口。

    Geometry fitter path must not call records_io write APIs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    called = {"count": 0}

    def _forbidden_call(*args: Any, **kwargs: Any) -> None:
        called["count"] += 1
        raise AssertionError("records_io write API must not be called by geometry fitter")

    monkeypatch.setattr(records_io, "write_json", _forbidden_call)
    monkeypatch.setattr(records_io, "write_artifact_json", _forbidden_call)
    monkeypatch.setattr(records_io, "copy_file_controlled", _forbidden_call)

    extractor = GeometryAlignInvarianceExtractor("geometry_align_invariance_sd3_v1", "v2", "a" * 64)
    cfg = {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "enable_latent_sync": True,
                "enable_align_invariance": True,
                "align_min_inlier_ratio": 0.2,
            }
        },
    }
    latents = np.random.RandomState(12).randn(1, 4, 8, 8).astype(np.float32)

    evidence = extractor.extract(cfg, inputs={"pipeline": _Pipeline(), "latents": latents})
    assert evidence.get("status") in {"ok", "fail", "mismatch", "absent"}
    assert called["count"] == 0
