"""
attack protocol append-only 扩展加载回归测试
"""

from __future__ import annotations

from main.evaluation import attack_plan
from main.evaluation import protocol_loader


def test_attack_protocol_append_only_versions_loadable() -> None:
    """
    功能：验证新增 family/params_version 可被协议加载器读取且计划顺序稳定。

    Verify append-only protocol families and params versions are loadable with deterministic ordering.

    Args:
        None.

    Returns:
        None.
    """
    protocol_spec = protocol_loader.load_attack_protocol_spec({})

    families = protocol_spec.get("families", {})
    params_versions = protocol_spec.get("params_versions", {})

    assert "jpeg" in families
    assert "gaussian_noise" in families
    assert "gaussian_blur" in families
    assert "composite" in families

    assert "jpeg::v1" in params_versions
    assert "gaussian_noise::v1" in params_versions
    assert "gaussian_blur::v1" in params_versions
    assert "composite::rotate_resize_v1" in params_versions

    generated_plan = attack_plan.generate_attack_plan(protocol_spec)
    assert generated_plan.conditions == sorted(generated_plan.conditions)
    assert "composite::rotate_resize_v1" in generated_plan.conditions
