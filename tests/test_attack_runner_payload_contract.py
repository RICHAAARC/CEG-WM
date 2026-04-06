"""
File purpose: Validate attack runner payload restoration contracts.
Module type: General module
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
from PIL import Image

from main.evaluation import attack_runner


def _build_test_image_array() -> np.ndarray:
    """
    功能：构造稳定的 RGB 图像数组夹具。 

    Build a deterministic RGB image array fixture.

    Args:
        None.

    Returns:
        RGB image array with shape [24, 24, 3].
    """
    return np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)


def test_gaussian_noise_preserves_pil_payload_type() -> None:
    """
    功能：验证 gaussian_noise 不会把 PIL 输入泄漏成 ndarray。 

    Ensure gaussian_noise preserves PIL payload type for image inputs.

    Args:
        None.

    Returns:
        None.
    """
    payload = Image.fromarray(_build_test_image_array(), mode="RGB")
    attack_spec: Dict[str, Any] = {
        "attack_family": "gaussian_noise",
        "params_version": "v1",
        "params": {"sigma": 0.03},
        "seed": 123,
    }

    result = attack_runner.apply_attack_transform(payload, attack_spec, random.Random(123))

    attacked_payload = result.get("payload")
    assert isinstance(attacked_payload, Image.Image)
    assert attacked_payload.mode == payload.mode
    attacked_array = np.asarray(attacked_payload)
    assert attacked_array.shape == np.asarray(payload).shape
    assert not np.array_equal(np.asarray(payload), attacked_array)


def test_gaussian_noise_preserves_ndarray_payload_type() -> None:
    """
    功能：验证 gaussian_noise 继续保持 ndarray 输入输出契约。 

    Ensure gaussian_noise preserves ndarray payload type for ndarray inputs.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_test_image_array()
    attack_spec: Dict[str, Any] = {
        "attack_family": "gaussian_noise",
        "params_version": "v1",
        "params": {"sigma": 0.03},
        "seed": 321,
    }

    result = attack_runner.apply_attack_transform(payload, attack_spec, random.Random(321))

    attacked_payload = result.get("payload")
    assert isinstance(attacked_payload, np.ndarray)
    assert attacked_payload.dtype == payload.dtype
    assert attacked_payload.shape == payload.shape
    assert not np.array_equal(payload, attacked_payload)