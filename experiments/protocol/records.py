"""定义论文候选实验 records 的最小通用结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from experiments.protocol.comparison import METHOD_ROLES


_DIGEST_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_EXECUTION_STATUSES = frozenset({"success", "failed", "excluded"})


@dataclass(frozen=True)
class ExperimentRecord:
    """表示一条可被论文产物重建流程消费的实验记录。"""

    record_id: str
    run_id: str
    comparison_group_name: str
    comparison_protocol_digest: str
    sample_manifest_digest: str
    split: str
    method_name: str
    method_role: str
    method_config_digest: str
    method_code_revision: str
    model_revision: str
    seed: int
    metric_name: str
    metric_value: float | None
    execution_status: str
    failure_reason: str | None
    exclusion_reason: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转为普通字典, 便于写入 JSONL、测试 fixture 或 artifact builder。"""
        return asdict(self)


REQUIRED_RECORD_FIELDS = (
    "record_id",
    "run_id",
    "comparison_group_name",
    "comparison_protocol_digest",
    "sample_manifest_digest",
    "split",
    "method_name",
    "method_role",
    "method_config_digest",
    "method_code_revision",
    "model_revision",
    "seed",
    "metric_name",
    "metric_value",
    "execution_status",
    "failure_reason",
    "exclusion_reason",
)


def validate_record(record: dict[str, Any]) -> list[str]:
    """返回缺失或不满足 provenance 约束的 record 字段列表。"""
    violations = [field_name for field_name in REQUIRED_RECORD_FIELDS if field_name not in record]
    if violations:
        return violations

    for field_name in (
        "comparison_protocol_digest",
        "sample_manifest_digest",
        "method_config_digest",
    ):
        if not _DIGEST_PATTERN.fullmatch(str(record[field_name])):
            violations.append(f"{field_name}_invalid")
    for field_name in (
        "record_id",
        "run_id",
        "comparison_group_name",
        "split",
        "method_name",
        "method_role",
        "method_code_revision",
        "model_revision",
        "metric_name",
    ):
        if not str(record[field_name]).strip():
            violations.append(f"{field_name}_missing")
    if record["method_role"] not in METHOD_ROLES:
        violations.append("method_role_invalid")
    if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
        violations.append("seed_invalid")

    execution_status = record["execution_status"]
    if execution_status not in _EXECUTION_STATUSES:
        violations.append("execution_status_invalid")
    metric_value = record["metric_value"]
    if execution_status == "success":
        if (
            not isinstance(metric_value, (int, float))
            or isinstance(metric_value, bool)
            or not math.isfinite(metric_value)
        ):
            violations.append("successful_metric_value_invalid")
        if str(record["failure_reason"] or "").strip():
            violations.append("successful_record_failure_reason_forbidden")
        if str(record["exclusion_reason"] or "").strip():
            violations.append("successful_record_exclusion_reason_forbidden")
    if execution_status == "failed" and not str(record["failure_reason"] or "").strip():
        violations.append("failure_reason_missing")
    if execution_status == "failed" and str(record["exclusion_reason"] or "").strip():
        violations.append("failed_record_exclusion_reason_forbidden")
    if execution_status == "excluded" and not str(record["exclusion_reason"] or "").strip():
        violations.append("exclusion_reason_missing")
    if execution_status == "excluded" and str(record["failure_reason"] or "").strip():
        violations.append("excluded_record_failure_reason_forbidden")
    if execution_status in {"failed", "excluded"} and metric_value is not None:
        violations.append("non_success_metric_value_forbidden")
    return violations
