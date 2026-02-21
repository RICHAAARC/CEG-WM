"""
File purpose: Whitelist enforcement test for planner impl id.
Module type: General module
"""

import pytest

from main.core.errors import GateEnforcementError
from main.policy.runtime_whitelist import load_runtime_whitelist, assert_impl_allowed


def test_registry_whitelist_required_for_planner_impl() -> None:
    whitelist = load_runtime_whitelist()

    impl_identity = {
        "content_extractor_id": "content_baseline_identity_v1",
        "geometry_extractor_id": "geometry_baseline_identity_v1",
        "fusion_rule_id": "fusion_baseline_identity_v1",
        "subspace_planner_id": "subspace_planner_not_whitelisted_v1",
        "sync_module_id": "geometry_sync_baseline_v1"
    }

    with pytest.raises(GateEnforcementError):
        assert_impl_allowed(whitelist, impl_identity)
