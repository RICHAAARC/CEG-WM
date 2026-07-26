"""定义外部 baseline 公平对比所需的共享协议与运行前批准。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re


_DIGEST_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
METHOD_ROLES = frozenset({"project_method", "external_baseline"})


@dataclass(frozen=True)
class ComparisonMethodSpec:
    """固定一个参与对比的方法实现、配置与已声明偏差。"""

    method_name: str
    method_role: str
    implementation_revision: str
    config_digest: str
    declared_deviation: str


@dataclass(frozen=True)
class ComparisonProtocol:
    """固定所有方法必须共享或预先声明的对比条件。"""

    comparison_group_name: str
    sample_manifest_digest: str
    split_manifest_digest: str
    generation_conditions_digest: str
    seed_policy_digest: str
    output_specification_digest: str
    attack_matrix_digest: str
    metric_set_digest: str
    calibration_split: str
    evaluation_split: str
    tuning_budget_policy_digest: str
    compute_budget_policy_digest: str
    failure_policy_digest: str
    exclusion_policy_digest: str
    methods: tuple[ComparisonMethodSpec, ...]

    def digest(self) -> str:
        """返回对规范化协议内容稳定计算的 SHA-256 摘要。"""
        canonical = json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class PreflightApproval:
    """证明一次运行已通过公平对比协议的静态检查。"""

    comparison_group_name: str
    protocol_digest: str
    sample_manifest_digest: str


def validate_comparison_protocol(protocol: ComparisonProtocol) -> list[str]:
    """返回对比协议中会使结果不可比较的问题代码。"""
    violations: list[str] = []
    digest_fields = (
        "sample_manifest_digest",
        "split_manifest_digest",
        "generation_conditions_digest",
        "seed_policy_digest",
        "output_specification_digest",
        "attack_matrix_digest",
        "metric_set_digest",
        "tuning_budget_policy_digest",
        "compute_budget_policy_digest",
        "failure_policy_digest",
        "exclusion_policy_digest",
    )
    if not protocol.comparison_group_name.strip():
        violations.append("comparison_group_name_missing")
    for field_name in digest_fields:
        if not _DIGEST_PATTERN.fullmatch(getattr(protocol, field_name)):
            violations.append(f"{field_name}_invalid")
    if not protocol.calibration_split.strip():
        violations.append("calibration_split_missing")
    if not protocol.evaluation_split.strip():
        violations.append("evaluation_split_missing")
    if protocol.calibration_split == protocol.evaluation_split:
        violations.append("calibration_and_evaluation_split_must_differ")

    method_names: set[str] = set()
    method_roles: set[str] = set()
    for method in protocol.methods:
        if not method.method_name.strip():
            violations.append("method_name_missing")
        elif method.method_name in method_names:
            violations.append("method_name_duplicate")
        method_names.add(method.method_name)
        method_roles.add(method.method_role)
        if method.method_role not in METHOD_ROLES:
            violations.append("method_role_invalid")
        if not method.implementation_revision.strip():
            violations.append("implementation_revision_missing")
        if not _DIGEST_PATTERN.fullmatch(method.config_digest):
            violations.append("method_config_digest_invalid")
    if "project_method" not in method_roles:
        violations.append("project_method_missing")
    if "external_baseline" not in method_roles:
        violations.append("external_baseline_missing")
    return violations


def approve_comparison_protocol(protocol: ComparisonProtocol) -> PreflightApproval:
    """校验协议并签发 runner 可消费的运行前批准。"""
    violations = validate_comparison_protocol(protocol)
    if violations:
        raise ValueError(", ".join(violations))
    return PreflightApproval(
        comparison_group_name=protocol.comparison_group_name,
        protocol_digest=protocol.digest(),
        sample_manifest_digest=protocol.sample_manifest_digest,
    )
