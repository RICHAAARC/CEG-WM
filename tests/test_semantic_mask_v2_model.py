"""
File purpose: 语义掩码 v2 双后端接口回归测试。
Module type: General module
"""

from pathlib import Path

import numpy as np
import pytest

from main.watermarking.content_chain.semantic_mask_provider import (
    build_semantic_saliency_mask_v2,
    _normalize_semantic_model_source,
)


def test_semantic_model_source_normalization() -> None:
    assert _normalize_semantic_model_source("basnet") == "basnet"
    assert _normalize_semantic_model_source("offline_heuristic") == "basnet"
    assert _normalize_semantic_model_source("inspyrenet") == "inspyrenet"
    assert _normalize_semantic_model_source("unknown_value") == "basnet"


def test_semantic_mask_v2_supports_basnet_and_inspyrenet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class DummyModel:
        def __init__(self, output_key: str) -> None:
            self._output_key = output_key

        def eval(self):
            return self

        def __call__(self, image_tensor):
            batch_size, _, height, width = image_tensor.shape
            payload = torch.ones((batch_size, 1, height, width), dtype=torch.float32)
            return {self._output_key: payload}

    def _fake_torch_load(path: str, map_location: str = "cpu", **kwargs):
        if "inspy" in path.lower():
            return DummyModel("pred")
        return DummyModel("saliency")

    monkeypatch.setattr(torch, "load", _fake_torch_load)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cfg = {}

    basnet_model_path = tmp_path / "basnet_dummy.pt"
    basnet_model_path.write_bytes(b"dummy")
    basnet_params = {
        "semantic_model_source": "basnet",
        "semantic_model_path": str(basnet_model_path),
        "semantic_model_version": "v2",
        "semantic_weights_id": "basnet_weights",
        "semantic_thresholding": "quantile",
        "saliency_threshold_quantile": 0.8,
        "open_iters": 0,
        "close_iters": 0,
        "semantic_preprocess": "rgb_normalized",
    }
    _, _, basnet_stats, _ = build_semantic_saliency_mask_v2(image, image.shape, cfg, basnet_params)

    inspy_model_path = tmp_path / "inspy_dummy.pt"
    inspy_model_path.write_bytes(b"dummy")
    inspy_params = dict(basnet_params)
    inspy_params["semantic_model_source"] = "inspyrenet"
    inspy_params["semantic_model_path"] = str(inspy_model_path)
    _, _, inspy_stats, _ = build_semantic_saliency_mask_v2(image, image.shape, cfg, inspy_params)

    assert basnet_stats["model_artifact_anchor"]["model_source"] == "basnet"
    assert inspy_stats["model_artifact_anchor"]["model_source"] == "inspyrenet"
