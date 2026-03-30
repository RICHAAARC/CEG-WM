"""
文件目的：验证 runtime whitelist 与 policy semantics 的早期版本强绑定入口。
Module type: General module
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from main.core.errors import WhitelistSemanticsMismatchError
from main.policy.runtime_whitelist import (
    PolicyPathSemantics,
    RuntimeWhitelist,
    assert_consistent_with_semantics,
)


def _build_runtime_whitelist(*, version: str) -> RuntimeWhitelist:
    """
    功能：构造 runtime whitelist 测试夹具。

    Build the minimal runtime whitelist fixture used by the binding tests.

    Args:
        version: Runtime whitelist version string.

    Returns:
        RuntimeWhitelist fixture.
    """
    if not isinstance(version, str) or not version:
        raise TypeError("version must be non-empty str")

    whitelist_data: Dict[str, Any] = {
        "policy_path": {"allowed": ["content_only", "content_np_geo_rescue"]},
        "consistency_rules": {
            "policy_path_semantics_must_be_subset_of_whitelist": False,
            "require_semantics_exact_cover": False,
        },
    }
    return RuntimeWhitelist(
        data=whitelist_data,
        whitelist_version=version,
        whitelist_digest="whitelist_digest",
        whitelist_file_sha256="whitelist_file_sha256",
        whitelist_canon_sha256="whitelist_canon_sha256",
        whitelist_bound_digest="whitelist_bound_digest",
    )


def _build_policy_path_semantics(*, version: str) -> PolicyPathSemantics:
    """
    功能：构造 policy semantics 测试夹具。

    Build the minimal policy-path semantics fixture used by the binding tests.

    Args:
        version: Policy semantics version string.

    Returns:
        PolicyPathSemantics fixture.
    """
    if not isinstance(version, str) or not version:
        raise TypeError("version must be non-empty str")

    semantics_data: Dict[str, Any] = {
        "policy_paths": {
            "content_only": {},
            "content_np_geo_rescue": {},
        }
    }
    return PolicyPathSemantics(
        data=semantics_data,
        policy_path_semantics_version=version,
        policy_path_semantics_digest="policy_path_semantics_digest",
        policy_path_semantics_file_sha256="policy_path_semantics_file_sha256",
        policy_path_semantics_canon_sha256="policy_path_semantics_canon_sha256",
        policy_path_semantics_bound_digest="policy_path_semantics_bound_digest",
    )


def test_assert_consistent_with_semantics_passes_when_versions_and_policy_paths_match() -> None:
    """
    功能：版本一致且 policy_path 闭包一致时必须通过早期一致性校验。

    Verify the early binding check succeeds when version binding and
    policy-path closure both remain consistent.

    Returns:
        None.
    """
    whitelist = _build_runtime_whitelist(version="v2.6")
    semantics = _build_policy_path_semantics(version="v2.6")

    assert_consistent_with_semantics(whitelist, semantics)


def test_assert_consistent_with_semantics_raises_when_versions_mismatch() -> None:
    """
    功能：版本不一致时必须在早期一致性校验入口报错。

    Verify the early binding check fails fast when whitelist and semantics
    versions diverge.

    Returns:
        None.
    """
    whitelist = _build_runtime_whitelist(version="v2.4")
    semantics = _build_policy_path_semantics(version="v2.6")

    with pytest.raises(WhitelistSemanticsMismatchError, match="versions must match"):
        assert_consistent_with_semantics(whitelist, semantics)