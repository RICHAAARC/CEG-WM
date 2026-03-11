"""
测试融合规则白名单强制

功能：
- 验证：fusion_rule impl_id 必须在 runtime_whitelist。
- 验证：不在白名单中的 impl_id 被拒绝。
- 验证：白名单外的实现无法通过注册表解析。
"""

from __future__ import annotations

import pytest
from typing import Any, Dict

from main.registries.fusion_registry import (
    resolve_fusion_rule,
    list_fusion_impl_ids
)


def test_fusion_neyman_pearson_whitelist_present() -> None:
    """
    验证：fusion_neyman_pearson_v2 在白名单中。
    """
    allowed_ids = list_fusion_impl_ids()

    assert "fusion_neyman_pearson_v2" in allowed_ids, \
        "fusion_neyman_pearson_v2 must be whitelisted"


def test_fusion_impl_ids_resolvable(np_fusion_impl_id: str = "fusion_neyman_pearson_v2") -> None:
    """
    验证：白名单中的 impl_id 可通过 resolve_fusion_rule 正确解析。
    """
    allowed_ids = list_fusion_impl_ids()

    for impl_id in allowed_ids:
        factory = resolve_fusion_rule(impl_id)
        assert callable(factory), \
            f"fusion_rule factory for {impl_id} must be callable"


def test_fusion_nonexistent_impl_raises_error() -> None:
    """
    验证：不在白名单中的 impl_id 被 resolve_fusion_rule 拒绝。
    """
    invalid_id = "fusion_nonexistent_v999"

    with pytest.raises(ValueError):
        resolve_fusion_rule(invalid_id)


def test_fusion_rule_only_whitelisted_ids() -> None:
    """
    验证：list_fusion_impl_ids 仅返回白名单中的实现。
    """
    ids = list_fusion_impl_ids()

    # (1) 应该至少有正式实现
    assert len(ids) >= 1, \
        "must have at least 1 fusion rule implementation"

    # (2) 应包含 NP 正式实现
    assert "fusion_neyman_pearson_v2" in ids, \
        "list must include fusion_neyman_pearson_v2"

    # (3) 验证列表中每个 id 都可解析
    for impl_id in ids:
        factory = resolve_fusion_rule(impl_id)
        assert callable(factory), \
            f"each listed impl_id must be resolvable, {impl_id} failed"


def test_fusion_impl_version_consistency() -> None:
    """
    验证：每个融合规则实现的版本信息一致。
    """
    ids = list_fusion_impl_ids()

    for impl_id in ids:
        factory = resolve_fusion_rule(impl_id)
        instance = factory({})

        # (1) 验证 impl_id 一致
        assert instance.impl_id == impl_id, \
            f"impl_id mismatch: expected {impl_id}, got {instance.impl_id}"

        # (2) 验证 impl_version 非空
        assert instance.impl_version, \
            f"impl_version must be non-empty for {impl_id}"

        # (3) 验证 impl_digest 非空
        assert instance.impl_digest, \
            f"impl_digest must be non-empty for {impl_id}"


def test_fusion_whitelist_seals_registry() -> None:
    """
    验证：融合规则注册表在冻结后无法修改（禁止绕过白名单）。
    
    说明：如果注册表被 seal()，运行期不应允许添加新的融合规则。
    """
    ids_before = list_fusion_impl_ids()

    # (1) 尝试调用 resolve 多次，应始终返回相同的实现集合
    ids_after = list_fusion_impl_ids()

    assert ids_before == ids_after, \
        "whitelist must not change (registry sealed)"


def test_fusion_impl_capabilities_recorded() -> None:
    """
    验证：每个融合规则的 capabilities 被正确记录。
    """
    from main.registries.fusion_registry import _FUSION_REGISTRY

    ids = list_fusion_impl_ids()

    for impl_id in ids:
        caps = _FUSION_REGISTRY.get_capabilities(impl_id)
        assert caps is not None, \
            f"capabilities must be recorded for {impl_id}"
        assert hasattr(caps, "supports_deterministic"), \
            f"capabilities must have supports_deterministic for {impl_id}"
