"""
File purpose: 楠岃瘉鏂板琛屼负鍙傛暟鏈撼鍏ユ憳瑕佽緭鍏ュ煙鏃跺繀椤昏Е鍙?mismatch銆?
Module type: General module
"""

from __future__ import annotations

from main.watermarking.common import plan_digest_flow


def _build_subspace_result(topk_value: int) -> dict:
    return {
        "plan": {
            "planner_impl_identity": {
                "impl_id": "subspace_planner_v2",
                "impl_version": "v2",
                "impl_digest": "abc",
            },
            "verifiable_input_domain_spec": {
                "planner_input_digest": "planner_input_digest_v1",
            },
            "planner_params": {
                "k": 8,
                "topk": topk_value,
            },
        }
    }


def test_new_params_must_be_in_digest_or_fail() -> None:
    """
    鍔熻兘锛氭柊澧炲弬鏁板彉鍖栦絾娌跨敤鏃ф憳瑕佹椂锛屽繀椤昏Е鍙?mismatch銆?

    New behavior-affecting params must be reflected in digest domain.
    If params change while stale digest is reused, verification must fail.

    Args:
        None.

    Returns:
        None.
    """
    cfg = {"policy_path": "content_only"}

    _, plan_digest_v1, _, _ = plan_digest_flow.build_content_plan_and_digest(
        cfg,
        _build_subspace_result(topk_value=16),
        mask_digest="mask_digest_v1",
    )
    _, plan_digest_v2, _, _ = plan_digest_flow.build_content_plan_and_digest(
        cfg,
        _build_subspace_result(topk_value=24),
        mask_digest="mask_digest_v1",
    )

    assert isinstance(plan_digest_v1, str) and plan_digest_v1
    assert isinstance(plan_digest_v2, str) and plan_digest_v2
    assert plan_digest_v1 != plan_digest_v2

    status, reason = plan_digest_flow.verify_plan_digest(
        expected=plan_digest_v2,
        observed=plan_digest_v1,
    )
    assert status == "mismatch"
    assert reason == "plan_digest_mismatch"

