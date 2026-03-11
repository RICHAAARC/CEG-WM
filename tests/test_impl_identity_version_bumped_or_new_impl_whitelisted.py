"""
File purpose: 验证几何链升级采用 impl_version 演进或新 impl_id 白名单化。
Module type: General module
"""

from __future__ import annotations

from main.policy.runtime_whitelist import load_runtime_whitelist
from main.registries.geometry_registry import resolve_geometry_extractor


def test_impl_identity_version_bumped_or_new_impl_whitelisted() -> None:
    """
    功能：当前正式实现必须满足"版本提升或新增白名单"的演进约束。

    Upgraded implementation must either bump version or use new whitelisted impl_id.
    Verifies that the formal paper geometry extractor (attention_anchor_map_relation_v2)
    is present in the whitelist, replacing the deprecated attention_anchor_map_relation_v1.

    Args:
        None.

    Returns:
        None.
    """
    impl_id = "attention_anchor_map_relation_v2"
    factory = resolve_geometry_extractor(impl_id)
    instance = factory({})

    whitelist = load_runtime_whitelist()
    impl_cfg = whitelist.data.get("impl_id", {})
    allowed_flat = impl_cfg.get("allowed_flat", [])

    assert impl_id in allowed_flat
    assert hasattr(instance, "impl_version")
    assert isinstance(instance.impl_version, str)
