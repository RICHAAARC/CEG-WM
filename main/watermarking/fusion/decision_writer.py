"""
融合判决写入器

功能说明：
- 将 FusionDecision 写入 record。
- 依赖 ContractInterpretation 提供的 decision_field_path 定位写入位置。
- 包含输入校验，确保类型和必需字段的正确性。
- 设计为独立模块，专注于融合判决的写入逻辑。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import schema
from main.core.contracts import ContractInterpretation
from main.watermarking.fusion.interfaces import FusionDecision


def assert_decision_write_bypass_blocked(
    record: Dict[str, Any],
    interpretation: ContractInterpretation,
) -> None:
    """
    功能：阻断直接写 decision 的旁路路径。

    Block bypass path that writes decision field directly without fusion_result.

    Args:
        record: Record mapping.
        interpretation: Contract interpretation with decision field path.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If bypass pattern is detected.
    """
    if not isinstance(record, dict):
        # record 类型不合法，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不合法，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")

    decision_field_path = interpretation.records_schema.decision_field_path
    if not isinstance(decision_field_path, str) or not decision_field_path:
        # decision_field_path 不合法，必须 fail-fast。
        raise ValueError("decision_field_path must be non-empty str")

    found, decision_value = schema._get_value_by_field_path(record, decision_field_path)
    fusion_result = record.get("fusion_result")

    if found and decision_value is not None and fusion_result is None:
        # 检测到 decision 已被直接写入但 fusion_result 缺失，必须阻断。
        raise ValueError(
            "decision write bypass detected: decision exists without fusion_result"
        )


def apply_fusion_decision_to_record(
    record: Dict[str, Any],
    decision: FusionDecision,
    interpretation: ContractInterpretation
) -> None:
    """
    功能：将 FusionDecision 写入 record。

    Apply fusion decision to record at the contract-defined decision field path.

    Args:
        record: Record mapping to mutate.
        decision: FusionDecision instance.
        interpretation: Contract interpretation providing decision field path.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If decision field path is invalid.
    """
    if not isinstance(record, dict):
        # record 类型不合法，必须 fail-fast。
        raise TypeError("record must be dict")
    if not isinstance(decision, FusionDecision):
        # decision 类型不合法，必须 fail-fast。
        raise TypeError("decision must be FusionDecision")
    if not isinstance(interpretation, ContractInterpretation):
        # interpretation 类型不合法，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation")

    decision_field_path = interpretation.records_schema.decision_field_path
    if not isinstance(decision_field_path, str) or not decision_field_path:
        # decision_field_path 不合法，必须 fail-fast。
        raise ValueError("decision_field_path must be non-empty str")

    schema._set_value_by_field_path(record, decision_field_path, decision.is_watermarked)
