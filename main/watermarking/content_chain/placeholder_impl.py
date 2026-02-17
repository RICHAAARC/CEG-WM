"""
File purpose: Content chain placeholder extractor implementation.
Module type: General module

功能说明：
- 提供内容链占位 extractor，实现接口级可复算与可审计输出。
- 默认输出 absent 语义，确保不引入统计口径污染。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from main.core import digests

from .interfaces import ContentEvidence


TRACE_VERSION = "v1"
DEFAULT_FAILURE_REASON = "placeholder_absent"


class ContentEvidencePlaceholderExtractor:
    """
    功能：内容链占位 extractor。

    Placeholder extractor emitting deterministic absent evidence.

    Args:
        impl_identity: Implementation identity string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If any input is invalid.
    """

    def __init__(self, impl_identity: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_identity, str) or not impl_identity:
            # impl_identity 输入不合法，必须 fail-fast。
            raise ValueError("impl_identity must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 输入不合法，必须 fail-fast。
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 输入不合法，必须 fail-fast。
            raise ValueError("impl_digest must be non-empty str")
        self.impl_identity = impl_identity
        self.impl_version = impl_version
        self.impl_digest = impl_digest

    def extract(
        self,
        cfg: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None
    ) -> ContentEvidence:
        """
        功能：输出占位内容链证据。

        Emit placeholder content evidence with deterministic audit fields.

        Args:
            cfg: Config mapping.
            inputs: Optional input mapping for future compatibility.

        Returns:
            ContentEvidence instance with absent status.

        Raises:
            TypeError: If cfg or inputs types are invalid.
        """
        if not isinstance(cfg, dict):
            # cfg 类型不合法，必须 fail-fast。
            raise TypeError("cfg must be dict")
        if inputs is not None and not isinstance(inputs, dict):
            # inputs 类型不合法，必须 fail-fast。
            raise TypeError("inputs must be dict or None")

        trace_payload = _build_trace_payload(
            cfg,
            inputs,
            self.impl_identity,
            self.impl_version,
            self.impl_digest
        )
        trace_digest = digests.canonical_sha256(trace_payload)

        audit = {
            "impl_identity": self.impl_identity,
            "impl_version": self.impl_version,
            "impl_digest": self.impl_digest,
            "trace_digest": trace_digest
        }

        return ContentEvidence(
            status="absent",
            score=None,
            audit=audit,
            mask_digest=None,
            mask_stats=None,
            plan_digest=None,
            basis_digest=None,
            lf_trace_digest=None,
            hf_trace_digest=None,
            lf_score=None,
            hf_score=None,
            score_parts=None,
            content_failure_reason=DEFAULT_FAILURE_REASON
        )


def _build_trace_payload(
    cfg: Dict[str, Any],
    inputs: Optional[Dict[str, Any]],
    impl_identity: str,
    impl_version: str,
    impl_digest: str
) -> Dict[str, Any]:
    """
    功能：构造可复算的 trace digest 输入域。

    Build deterministic trace payload for digest computation.

    Args:
        cfg: Config mapping.
        inputs: Optional inputs mapping.
        impl_identity: Implementation identity string.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        JSON-like payload mapping.

    Raises:
        TypeError: If any input types are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if inputs is not None and not isinstance(inputs, dict):
        # inputs 类型不合法，必须 fail-fast。
        raise TypeError("inputs must be dict or None")
    if not isinstance(impl_identity, str) or not impl_identity:
        # impl_identity 类型不合法，必须 fail-fast。
        raise TypeError("impl_identity must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 类型不合法，必须 fail-fast。
        raise TypeError("impl_version must be non-empty str")
    if not isinstance(impl_digest, str) or not impl_digest:
        # impl_digest 类型不合法，必须 fail-fast。
        raise TypeError("impl_digest must be non-empty str")

    normalized_inputs = inputs or {}
    payload = {
        "trace_version": TRACE_VERSION,
        "impl_identity": impl_identity,
        "impl_version": impl_version,
        "impl_digest": impl_digest,
        "cfg_keys": _sorted_keys(cfg, "cfg"),
        "inputs_keys": _sorted_keys(normalized_inputs, "inputs"),
        "inputs_present": inputs is not None
    }
    return payload


def _sorted_keys(mapping: Dict[str, Any], field_name: str) -> List[str]:
    """
    功能：返回稳定排序的 key 列表。

    Return sorted keys for deterministic digest input.

    Args:
        mapping: Mapping to list keys from.
        field_name: Field name for error context.

    Returns:
        Sorted list of keys.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不合法，必须 fail-fast。
        raise TypeError(f"{field_name} must be dict")
    if not isinstance(field_name, str) or not field_name:
        # field_name 类型不合法，必须 fail-fast。
        raise TypeError("field_name must be non-empty str")
    for key in mapping.keys():
        if not isinstance(key, str):
            # key 类型不合法，必须 fail-fast。
            raise TypeError(f"{field_name} keys must be str")
    return sorted(mapping.keys())
