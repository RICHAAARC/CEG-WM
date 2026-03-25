"""
File purpose: Content chain baseline extractor tests.
Module type: General module

功能说明：
- 验证内容链基线 extractor 的 absent 语义与审计字段。
- 验证 extract(cfg) 兼容入口与可复算 trace_digest。
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from main.watermarking.content_chain.content_baseline_extractor import (
    ContentEvidenceBaselineExtractor
)


def _block_write(*args, **kwargs):
    """
    功能：阻断写盘调用。

    Block write attempts to ensure no disk writes occur.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        None.

    Raises:
        RuntimeError: Always raised to indicate a write attempt.
    """
    if not isinstance(args, tuple):
        # args 类型不合法，必须 fail-fast。
        raise TypeError("args must be tuple")
    if not isinstance(kwargs, dict):
        # kwargs 类型不合法，必须 fail-fast。
        raise TypeError("kwargs must be dict")
    raise RuntimeError("write blocked")


def test_content_baseline_extractor_absent_and_audit(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    功能：验证基线 extractor 的 absent 语义与审计字段。

    Verify baseline extractor absent semantics and audit fields.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(Path, "write_text", _block_write)
    monkeypatch.setattr(Path, "write_bytes", _block_write)
    original_path_open = Path.open

    def _blocked_path_open(self, *args, **kwargs):
        if not isinstance(self, Path):
            # self 类型不合法，必须 fail-fast。
            raise TypeError("self must be Path")
        if not isinstance(args, tuple):
            # args 类型不合法，必须 fail-fast。
            raise TypeError("args must be tuple")
        if not isinstance(kwargs, dict):
            # kwargs 类型不合法，必须 fail-fast。
            raise TypeError("kwargs must be dict")
        mode = "r"
        if args:
            mode = args[0]
        else:
            mode = kwargs.get("mode", "r")
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise RuntimeError("write blocked")
        return original_path_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _blocked_path_open)

    extractor = ContentEvidenceBaselineExtractor(
        impl_identity="content_baseline_v1",
        impl_version="v1",
        impl_digest="digest_baseline"
    )
    cfg: Dict[str, Any] = {"policy_path": "test_policy"}

    evidence = extractor.extract(cfg)
    evidence_dict = evidence.as_dict()

    assert evidence_dict["status"] == "absent"
    assert evidence_dict["score"] is None
    assert evidence_dict["content_failure_reason"] == "evidence_absent"

    audit = evidence_dict["audit"]
    assert set(audit.keys()) == {
        "impl_identity",
        "impl_version",
        "impl_digest",
        "trace_digest"
    }
    assert audit["impl_identity"] == "content_baseline_v1"
    assert audit["impl_version"] == "v1"
    assert audit["impl_digest"] == "digest_baseline"
    assert isinstance(audit["trace_digest"], str)

    evidence_repeat = extractor.extract(cfg)
    assert evidence_repeat.audit["trace_digest"] == audit["trace_digest"]
