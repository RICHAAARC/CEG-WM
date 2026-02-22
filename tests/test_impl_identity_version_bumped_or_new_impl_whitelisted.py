"""
File purpose: 验证几何链升级采用 impl_version 演进或新 impl_id 白名单化。
Module type: General module
"""

from __future__ import annotations

from main.policy.runtime_whitelist import load_runtime_whitelist
from main.registries.geometry_registry import resolve_geometry_extractor


def test_impl_identity_version_bumped_or_new_impl_whitelisted() -> None:
    """
    功能：当前实现必须满足“版本提升或新增白名单”的演进约束。

    Upgraded implementation must either bump version or use new whitelisted impl_id.

    Args:
        None.

    Returns:
        None.
    """
    impl_id = "geometry_align_invariance_sd3_v1"
    factory = resolve_geometry_extractor(impl_id)
    instance = factory({})

    whitelist = load_runtime_whitelist()
    impl_cfg = whitelist.data.get("impl_id", {})
    allowed_flat = impl_cfg.get("allowed_flat", [])

    assert impl_id in allowed_flat
    assert hasattr(instance, "impl_version")
    assert isinstance(instance.impl_version, str)
    assert instance.impl_version in {"v2", "v1"}
    assert instance.impl_version == "v2"
