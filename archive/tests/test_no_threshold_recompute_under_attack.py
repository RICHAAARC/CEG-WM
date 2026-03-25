"""
File purpose: 攻击执行阶段阈值不重估回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Any

import pytest

from main.evaluation import attack_runner
from main.evaluation import protocol_loader


def test_no_threshold_recompute_under_attack(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：验证攻击执行阶段不会触发阈值重估函数。

    Assert attack execution does not invoke NP threshold recomputation functions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    def _raise_if_called(*args: Any, **kwargs: Any) -> None:
        _ = args
        _ = kwargs
        raise AssertionError("NP threshold recompute must not be called in attack execution")

    monkeypatch.setattr(
        "main.watermarking.fusion.neyman_pearson.compute_thresholds_digest",
        _raise_if_called,
    )
    monkeypatch.setattr(
        "main.watermarking.fusion.neyman_pearson.compute_threshold_metadata_digest",
        _raise_if_called,
    )

    protocol_spec = protocol_loader.load_attack_protocol_spec({})
    condition_spec = attack_runner.resolve_condition_spec_from_protocol(protocol_spec, "rotate::v1")

    result = attack_runner.run_attacks_for_condition(
        images_or_latents={"sample": [1, 2, 3]},
        condition_spec=condition_spec,
        seed=7,
    )

    assert result.attack_status == "ok"
    assert isinstance(result.attack_trace_digest, str)
    assert len(result.attack_trace_digest) == 64
