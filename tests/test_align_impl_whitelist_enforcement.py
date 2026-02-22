"""
File purpose: 验证 align impl 未在 whitelist 时必须拒绝。
Module type: General module
"""

from __future__ import annotations

import copy
from typing import Dict

import pytest

from main.core.errors import GateEnforcementError
from main.policy.runtime_whitelist import RuntimeWhitelist, assert_impl_allowed, load_runtime_whitelist


def test_align_impl_must_be_whitelisted() -> None:
    """
    功能：geometry_align_invariance_sd3_v1 未被允许时必须触发 GateEnforcementError。

    Must fail when geometry align impl is not in whitelist.

    Args:
        None.

    Returns:
        None.
    """
    whitelist = load_runtime_whitelist()
    mutated = copy.deepcopy(whitelist.data)

    impl_cfg = mutated.get("impl_id", {})
    allowed_by_domain = impl_cfg.get("allowed_by_domain", {})
    geometry_allowed = allowed_by_domain.get("geometry_extractor", [])
    geometry_allowed = [item for item in geometry_allowed if item != "geometry_align_invariance_sd3_v1"]
    allowed_by_domain["geometry_extractor"] = geometry_allowed

    allowed_flat = impl_cfg.get("allowed_flat", [])
    impl_cfg["allowed_flat"] = [item for item in allowed_flat if item != "geometry_align_invariance_sd3_v1"]
    impl_cfg["allowed_by_domain"] = allowed_by_domain
    mutated["impl_id"] = impl_cfg

    restricted = RuntimeWhitelist(
        data=mutated,
        whitelist_version=whitelist.whitelist_version,
        whitelist_digest=whitelist.whitelist_digest,
        whitelist_file_sha256=whitelist.whitelist_file_sha256,
        whitelist_canon_sha256=whitelist.whitelist_canon_sha256,
        whitelist_bound_digest=whitelist.whitelist_bound_digest,
    )

    impl_identity: Dict[str, str] = {
        "content_extractor_id": "unified_content_extractor_v1",
        "geometry_extractor_id": "geometry_align_invariance_sd3_v1",
        "fusion_rule_id": "fusion_baseline_identity_v1",
        "subspace_planner_id": "subspace_planner_v1",
        "sync_module_id": "geometry_sync_baseline_v1",
    }

    with pytest.raises(GateEnforcementError):
        assert_impl_allowed(restricted, impl_identity)
