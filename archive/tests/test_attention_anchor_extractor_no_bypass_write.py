"""
File purpose: 验证 AttentionAnchorExtractor 不会直接旁路写盘。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from main.core import records_io
from main.watermarking.geometry_chain.attention_anchor_extractor import (
    ATTENTION_ANCHOR_EXTRACTOR_ID,
    ATTENTION_ANCHOR_EXTRACTOR_VERSION,
    AttentionAnchorExtractor,
)


class _TransformerConfig:
    patch_size = 2


class _Transformer:
    config = _TransformerConfig()


class _Pipeline:
    transformer = _Transformer()


def test_attention_anchor_extractor_must_not_write_records_or_artifacts(monkeypatch) -> None:
    """
    功能：提取器执行期间不得直接调用 records_io 写盘入口。

    Extractor execution must not directly call records_io write APIs.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    called = {"count": 0}

    def _forbidden_call(*args: Any, **kwargs: Any) -> None:
        called["count"] += 1
        raise AssertionError("records_io write API must not be called by extractor")

    monkeypatch.setattr(records_io, "write_json", _forbidden_call)
    monkeypatch.setattr(records_io, "write_artifact_json", _forbidden_call)
    monkeypatch.setattr(records_io, "copy_file_controlled", _forbidden_call)

    extractor = AttentionAnchorExtractor(
        ATTENTION_ANCHOR_EXTRACTOR_ID,
        ATTENTION_ANCHOR_EXTRACTOR_VERSION,
        "a" * 64,
    )

    cfg = {
        "model_id": "stabilityai/stable-diffusion-3-medium",
        "inference_height": 512,
        "inference_width": 512,
        "detect": {
            "geometry": {
                "enabled": True,
                "enable_attention_anchor": True,
                "anchor_top_k": 4,
            }
        },
    }
    latents = np.random.RandomState(13).randn(1, 4, 8, 8).astype(np.float32)

    evidence = extractor.extract(cfg, inputs={"pipeline": _Pipeline(), "latents": latents})
    assert evidence.get("status") == "ok"
    assert called["count"] == 0
