"""
File purpose: plan_digest 数据流收口工具。
Module type: General module

功能说明：
- 统一构建 plan 与输入锚点摘要，确保摘要口径单一且可复算。
- 提供 expected/observed plan_digest 的一致性判定，输出冻结语义 ok/mismatch/absent。
- 提供 record 绑定函数，将 plan 锚点字段以 append-only 方式写入记录。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from main.core import digests


PLAN_INPUT_SCHEMA_VERSION = "v2"


def build_content_plan_and_digest(
    cfg: Dict[str, Any],
    subspace_result: Any,
    mask_digest: Optional[str],
    *,
    mask_binding: Optional[Dict[str, Any]] = None,
    mask_params_digest: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str], str, Dict[str, Any]]:
    """
    功能：构建内容链 plan 与摘要锚点。 

    Build content plan object and digest anchors from planner result.

    Args:
        cfg: Configuration mapping.
        subspace_result: Planner result object or mapping.
        mask_digest: Optional semantic mask digest.

    Returns:
        Tuple of (plan_obj, plan_digest, plan_input_digest, plan_meta).
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")

    plan_obj: Dict[str, Any] = {}
    if hasattr(subspace_result, "plan") and isinstance(subspace_result.plan, dict):
        plan_obj = dict(subspace_result.plan)
    elif isinstance(subspace_result, dict):
        plan_candidate = subspace_result.get("plan")
        if isinstance(plan_candidate, dict):
            plan_obj = dict(plan_candidate)

    plan_digest = None
    if hasattr(subspace_result, "plan_digest") and isinstance(subspace_result.plan_digest, str):
        plan_digest = subspace_result.plan_digest
    elif isinstance(subspace_result, dict):
        digest_candidate = subspace_result.get("plan_digest")
        if isinstance(digest_candidate, str) and digest_candidate:
            plan_digest = digest_candidate
    if plan_digest is None and plan_obj:
        plan_digest = digests.canonical_sha256(plan_obj)

    planner_impl_identity = None
    planner_input_digest = None
    planner_params_digest = None
    if isinstance(plan_obj, dict):
        planner_impl_identity = plan_obj.get("planner_impl_identity")
        verifiable_spec = plan_obj.get("verifiable_input_domain_spec")
        if isinstance(verifiable_spec, dict):
            planner_input_digest = verifiable_spec.get("planner_input_digest")
        planner_params = plan_obj.get("planner_params")
        if isinstance(planner_params, dict):
            planner_params_digest = digests.canonical_sha256(planner_params)

    payload = {
        "plan_input_schema_version": PLAN_INPUT_SCHEMA_VERSION,
        "mask_digest": mask_digest if isinstance(mask_digest, str) and mask_digest else "<absent>",
        "planner_impl_identity": planner_impl_identity,
        "planner_input_digest": planner_input_digest if isinstance(planner_input_digest, str) and planner_input_digest else "<absent>",
        "planner_params_digest": planner_params_digest if isinstance(planner_params_digest, str) and planner_params_digest else "<absent>",
        "policy_path": cfg.get("policy_path", "<absent>"),
        "mask_binding": mask_binding if isinstance(mask_binding, dict) else "<absent>",
        "mask_params_digest": mask_params_digest if isinstance(mask_params_digest, str) and mask_params_digest else "<absent>",
    }
    plan_input_digest = digests.canonical_sha256(payload)
    plan_meta = {
        "plan_input_schema_version": PLAN_INPUT_SCHEMA_VERSION,
        "planner_input_digest": planner_input_digest if isinstance(planner_input_digest, str) else None,
        "planner_params_digest": planner_params_digest,
        "plan_input_payload": payload
    }
    return plan_obj, plan_digest, plan_input_digest, plan_meta


def verify_plan_digest(expected: Optional[str], observed: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    功能：校验 plan_digest 一致性并返回冻结语义状态。 

    Verify expected/observed plan digests and return frozen status semantics.

    Args:
        expected: Expected digest from bound input record/artifact.
        observed: Observed digest from detect-time evidence.

    Returns:
        Tuple of (status, reason) where status is one of ok/mismatch/absent.
    """
    expected_valid = isinstance(expected, str) and bool(expected)
    observed_valid = isinstance(observed, str) and bool(observed)

    if not expected_valid or not observed_valid:
        return "absent", "plan_digest_absent"
    if expected != observed:
        return "mismatch", "plan_digest_mismatch"
    return "ok", None


def bind_plan_to_record(
    record: Dict[str, Any],
    *,
    plan_obj: Dict[str, Any],
    plan_digest: Optional[str],
    plan_input_digest: str,
    plan_meta: Dict[str, Any]
) -> None:
    """
    功能：将 plan 锚点字段写入记录（append-only）。

    Bind plan anchors into record fields in an append-only manner.

    Args:
        record: Mutable record mapping.
        plan_obj: Structured plan mapping.
        plan_digest: Plan digest string or None.
        plan_input_digest: Digest of plan input anchors.
        plan_meta: Additional plan metadata mapping.

    Returns:
        None.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")
    if not isinstance(plan_obj, dict):
        raise TypeError("plan_obj must be dict")
    if not isinstance(plan_input_digest, str) or not plan_input_digest:
        raise TypeError("plan_input_digest must be non-empty str")
    if not isinstance(plan_meta, dict):
        raise TypeError("plan_meta must be dict")

    record["subspace_plan"] = plan_obj
    record["plan_digest"] = plan_digest
    record["plan_input_digest"] = plan_input_digest
    record["plan_input_schema_version"] = plan_meta.get("plan_input_schema_version", PLAN_INPUT_SCHEMA_VERSION)
