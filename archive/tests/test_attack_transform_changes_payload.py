"""
攻击变换真实生效回归测试
"""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import numpy.typing as npt

from main.evaluation import attack_runner


def _build_test_image() -> npt.NDArray[np.uint8]:
    """
    功能：构建稳定测试图像。

    Build deterministic test image payload.

    Args:
        None.

    Returns:
        Numpy uint8 image array with shape [32, 32, 3].
    """
    grid = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    return grid


def test_attack_transform_changes_payload() -> None:
    """
    功能：验证攻击变换会真实改变 payload。

    Ensure apply_attack_transform performs real mutation on image payload.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_test_image()
    attack_spec: Dict[str, Any] = {
        "attack_family": "gaussian_noise",
        "params_version": "v1",
        "params": {"sigma": 0.03},
        "seed": 123,
    }

    result = attack_runner.apply_attack_transform(payload, attack_spec, random.Random(123))

    attacked_payload = result.get("payload")
    assert isinstance(attacked_payload, np.ndarray)
    assert attacked_payload.shape == payload.shape
    assert not np.array_equal(payload, attacked_payload)
    assert isinstance(result.get("attack_trace_digest"), str)
    assert len(result.get("attack_trace_digest", "")) == 64
