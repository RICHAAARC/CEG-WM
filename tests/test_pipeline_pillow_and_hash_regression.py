"""
File purpose: Regressions for Pillow compatibility and synthetic artifact hash stability.
Module type: General module
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from main.core import digests
from main.diffusion.sd3 import pipeline_factory


def _build_rgb_test_array() -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(20260222)
    array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    array[0, 0] = [0, 0, 0]
    array[0, 1] = [1, 1, 1]
    array[0, 2] = [254, 254, 254]
    array[0, 3] = [255, 255, 255]
    return array


def test_pillow_fromarray_rgb_semantics_remain_identical() -> None:
    image_array = _build_rgb_test_array()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        image_old = Image.fromarray(image_array, mode="RGB")

    image_new = Image.fromarray(image_array)

    old_array = np.asarray(image_old)
    new_array = np.asarray(image_new)

    assert old_array.shape == new_array.shape
    assert old_array.dtype == new_array.dtype
    assert np.array_equal(old_array, new_array)


def test_synthetic_pipeline_artifact_sha256_is_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_env_fp_digest(_obj: dict[str, Any]) -> str:
        return "env_fp_digest_test"

    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "build_env_fingerprint",
        lambda: {"python_version": "test", "platform": "test", "sys_platform": "test", "executable": "test", "torch_version": "<absent>", "cuda_available": "<absent>"},
    )
    monkeypatch.setattr(
        pipeline_factory.env_fingerprint,
        "compute_env_fingerprint_canon_sha256",
        _mock_env_fp_digest,
    )

    cfg: dict[str, Any] = {
        "pipeline_impl_id": "sd3_diffusers_shell",
        "pipeline_build_enabled": True,
        "model": {"height": 32, "width": 32, "dtype": "float32"},
        "inference_num_steps": 4,
        "seed": 123,
    }

    result = pipeline_factory.build_pipeline_shell(cfg)
    pipeline_obj = result.get("pipeline_obj")
    assert pipeline_obj is not None

    output = pipeline_obj(
        prompt="hash stability",
        num_inference_steps=4,
        height=32,
        width=32,
        seed=123,
    )
    image = output.images[0]

    artifact_path = tmp_path / "synthetic_artifact.png"
    image.save(artifact_path, format="PNG", optimize=False, compress_level=0)

    artifact_sha256 = digests.file_sha256(artifact_path)
    assert artifact_sha256 == "1f293978f94fde3668e42288752f554ffed90863049f32aecf0adea7b3deec8e"
