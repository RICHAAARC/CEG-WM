"""
File purpose: 攻击随机种子控制回归测试。
Module type: General module
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
        Numpy uint8 image array.
    """
    return np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)


def test_attack_seed_control() -> None:
    """
    功能：验证随机种子变化会导致 attack trace digest 改变。

    Verify changing attack seed changes attack_trace_digest.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_test_image()

    spec_seed_1: Dict[str, Any] = {
        "attack_family": "gaussian_noise",
        "params_version": "v1",
        "params": {"sigma": 0.02},
        "seed": 1,
    }
    spec_seed_2: Dict[str, Any] = {
        "attack_family": "gaussian_noise",
        "params_version": "v1",
        "params": {"sigma": 0.02},
        "seed": 2,
    }

    result_seed_1 = attack_runner.apply_attack_transform(payload, spec_seed_1, random.Random(1))
    result_seed_2 = attack_runner.apply_attack_transform(payload, spec_seed_2, random.Random(2))

    assert result_seed_1["attack_trace_digest"] != result_seed_2["attack_trace_digest"]
