"""
攻击 trace digest 稳定性回归测试
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
    return np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)


def test_attack_digest_stable() -> None:
    """
    功能：验证相同参数与种子时 attack digest 稳定。

    Verify attack_trace_digest is deterministic under same payload/spec/seed.

    Args:
        None.

    Returns:
        None.
    """
    payload = _build_test_image()
    attack_spec: Dict[str, Any] = {
        "attack_family": "rotate",
        "params_version": "v1",
        "params": {"degrees": 10, "interpolation": "bilinear"},
        "seed": 2026,
    }

    first = attack_runner.apply_attack_transform(payload, attack_spec, random.Random(2026))
    second = attack_runner.apply_attack_transform(payload, attack_spec, random.Random(2026))

    assert first["attack_trace_digest"] == second["attack_trace_digest"]
    assert first["attack_digest"] == second["attack_digest"]
