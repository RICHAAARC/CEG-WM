"""
File purpose: 验证 decision 旁路写入被阻断。
Module type: General module
"""

from __future__ import annotations

import pytest

from main.core import schema
from main.watermarking.fusion import decision_writer


def test_decision_write_bypass_is_blocked(mock_interpretation) -> None:
    """
    功能：直接写 decision 且缺失 fusion_result 时必须失败。

    Direct decision write without fusion_result must be blocked.

    Args:
        mock_interpretation: Contract interpretation fixture.

    Returns:
        None.
    """
    record = {}
    decision_field_path = mock_interpretation.records_schema.decision_field_path
    schema._set_value_by_field_path(record, decision_field_path, True)

    with pytest.raises(ValueError, match="bypass"):
        decision_writer.assert_decision_write_bypass_blocked(record, mock_interpretation)


def test_decision_write_with_fusion_result_is_allowed(mock_interpretation) -> None:
    """
    功能：存在 fusion_result 时不应误报旁路。

    Bypass guard should not fail when fusion_result exists.

    Args:
        mock_interpretation: Contract interpretation fixture.

    Returns:
        None.
    """
    record = {"fusion_result": {"decision_status": "decided"}}
    decision_field_path = mock_interpretation.records_schema.decision_field_path
    schema._set_value_by_field_path(record, decision_field_path, True)

    decision_writer.assert_decision_write_bypass_blocked(record, mock_interpretation)
