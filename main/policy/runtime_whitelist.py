"""
运行期白名单与策略路径语义加载

功能说明：
- 运行期白名单与策略路径语义加载与校验模块。
"""

from pathlib import Path
from typing import Dict, Any, Set, List
from dataclasses import dataclass

from main.core import config_loader
from main.core import digests
from main.core.errors import (
    YAMLLoadError,
    MissingRequiredFieldError,
    WhitelistSemanticsMismatchError,
    GateEnforcementError,
    FrozenContractPathNotAuthoritativeError
)
from main.core.contracts import FrozenContracts, get_contract_interpretation
from main.registries import pipeline_registry
import os


def _is_test_environment() -> bool:
    """
    功能：判断当前是否在测试环境。
    
    Detect whether the current process is running under pytest.
    Uses PYTEST_CURRENT_TEST environment variable which is set during test execution.
    
    Returns:
        True if executing in test environment, False otherwise.
    """
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


@dataclass
class RuntimeWhitelist:
    """
    功能：运行期白名单加载结果。
    
    Parsed runtime whitelist with computed digests.
    
    Attributes:
        data: Full YAML object.
        whitelist_version: Version from YAML.
        whitelist_digest: Semantic digest.
        whitelist_file_sha256: SHA256 of raw file.
        whitelist_canon_sha256: SHA256 of canonical JSON.
        whitelist_bound_digest: Bound digest.
    """
    data: Dict[str, Any]
    whitelist_version: str
    whitelist_digest: str
    whitelist_file_sha256: str
    whitelist_canon_sha256: str
    whitelist_bound_digest: str


@dataclass
class PolicyPathSemantics:
    """
    功能：策略路径语义加载结果。
    
    Parsed policy path semantics with computed digests.
    
    Attributes:
        data: Full YAML object.
        policy_path_semantics_version: Version from YAML.
        policy_path_semantics_digest: Semantic digest.
        policy_path_semantics_file_sha256: SHA256 of raw file.
        policy_path_semantics_canon_sha256: SHA256 of canonical JSON.
        policy_path_semantics_bound_digest: Bound digest.
    """
    data: Dict[str, Any]
    policy_path_semantics_version: str
    policy_path_semantics_digest: str
    policy_path_semantics_file_sha256: str
    policy_path_semantics_canon_sha256: str
    policy_path_semantics_bound_digest: str


def load_runtime_whitelist(
    path: str = "configs/runtime_whitelist.yaml",
    *,
    allow_non_authoritative: bool = False
) -> RuntimeWhitelist:
    """
    功能：加载运行期白名单并计算所有 digest。
    
    Load runtime_whitelist.yaml and compute all digests.
    
    Args:
        path: Path to runtime_whitelist.yaml.
        allow_non_authoritative: Whether to allow non-authoritative paths for tests.
    
    Returns:
        RuntimeWhitelist object.
    
    Raises:
        YAMLLoadError: If loading fails.
        MissingRequiredFieldError: If whitelist_version is missing.
        FrozenContractPathNotAuthoritativeError: If non-authoritative path is used.
    """
    if not isinstance(path, str) or not path:
        # path 输入不合法，必须 fail-fast。
        raise TypeError("path must be non-empty str")
    if not isinstance(allow_non_authoritative, bool):
        # allow_non_authoritative 输入不合法，必须 fail-fast。
        raise TypeError("allow_non_authoritative must be bool")

    normalized_path = Path(path).as_posix()
    if normalized_path != "configs/runtime_whitelist.yaml":
        # 非权威路径例外仅测试环境可用
        is_test = _is_test_environment()
        if not (allow_non_authoritative and is_test):
            raise FrozenContractPathNotAuthoritativeError(
                "runtime_whitelist path is not authoritative",
                field_path="whitelist_source_path",
                actual_path=normalized_path
            )
    # 加载 YAML 与 provenance。
    obj, provenance = config_loader.load_yaml_with_provenance(path)
    
    # 解析 whitelist_version。
    whitelist_version = obj.get("whitelist_version")
    if not whitelist_version:
        raise MissingRequiredFieldError(
            f"whitelist_version missing in {path}"
        )
    
    # 计算 digest。
    whitelist_digest_value = digests.semantic_digest(obj)
    whitelist_file_sha256_value = provenance.file_sha256
    whitelist_canon_sha256_value = provenance.canon_sha256
    whitelist_bound_digest_value = digests.bound_digest(
        version=whitelist_version,
        semantic_digest_value=whitelist_digest_value,
        file_sha256_value=whitelist_file_sha256_value,
        canon_sha256_value=whitelist_canon_sha256_value
    )
    
    return RuntimeWhitelist(
        data=obj,
        whitelist_version=whitelist_version,
        whitelist_digest=whitelist_digest_value,
        whitelist_file_sha256=whitelist_file_sha256_value,
        whitelist_canon_sha256=whitelist_canon_sha256_value,
        whitelist_bound_digest=whitelist_bound_digest_value
    )


def load_policy_path_semantics(
    path: str = "configs/policy_path_semantics.yaml",
    *,
    allow_non_authoritative: bool = False
) -> PolicyPathSemantics:
    """
    功能：加载策略路径语义并计算所有 digest。
    
    Load policy_path_semantics.yaml and compute all digests.
    
    Args:
        path: Path to policy_path_semantics.yaml.
        allow_non_authoritative: Whether to allow non-authoritative paths for tests.
    
    Returns:
        PolicyPathSemantics object.
    
    Raises:
        YAMLLoadError: If loading fails.
        MissingRequiredFieldError: If policy_path_semantics_version is missing.
        FrozenContractPathNotAuthoritativeError: If non-authoritative path is used.
    """
    if not isinstance(path, str) or not path:
        # path 输入不合法，必须 fail-fast。
        raise TypeError("path must be non-empty str")
    if not isinstance(allow_non_authoritative, bool):
        # allow_non_authoritative 输入不合法，必须 fail-fast。
        raise TypeError("allow_non_authoritative must be bool")

    normalized_path = Path(path).as_posix()
    if normalized_path != "configs/policy_path_semantics.yaml":
        # 非权威路径例外仅测试环境可用
        is_test = _is_test_environment()
        if not (allow_non_authoritative and is_test):
            raise FrozenContractPathNotAuthoritativeError(
                "policy_path_semantics path is not authoritative",
                field_path="semantics_source_path",
                actual_path=normalized_path
            )
    # 加载 YAML 与 provenance。
    obj, provenance = config_loader.load_yaml_with_provenance(path)
    
    # 解析 policy_path_semantics_version。
    pps_version = obj.get("policy_path_semantics_version")
    if not pps_version:
        raise MissingRequiredFieldError(
            f"policy_path_semantics_version missing in {path}"
        )
    
    # 计算 digest。
    pps_digest_value = digests.semantic_digest(obj)
    pps_file_sha256_value = provenance.file_sha256
    pps_canon_sha256_value = provenance.canon_sha256
    pps_bound_digest_value = digests.bound_digest(
        version=pps_version,
        semantic_digest_value=pps_digest_value,
        file_sha256_value=pps_file_sha256_value,
        canon_sha256_value=pps_canon_sha256_value
    )
    
    return PolicyPathSemantics(
        data=obj,
        policy_path_semantics_version=pps_version,
        policy_path_semantics_digest=pps_digest_value,
        policy_path_semantics_file_sha256=pps_file_sha256_value,
        policy_path_semantics_canon_sha256=pps_canon_sha256_value,
        policy_path_semantics_bound_digest=pps_bound_digest_value
    )


def assert_consistent_with_semantics(
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：校验 whitelist 与 semantics 的一致性。
    
    Validate that whitelist.policy_path.allowed is consistent with semantics.policy_paths keys.
    
    Args:
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
    
    Raises:
        WhitelistSemanticsMismatchError: If consistency check fails.
    """
    # 提取 whitelist 中的 policy_path.allowed。
    policy_path_config = whitelist.data.get("policy_path", {})
    allowed_policy_paths = policy_path_config.get("allowed", [])
    allowed_set = set(allowed_policy_paths)
    
    # 提取 semantics 中的 policy_paths 键。
    policy_paths_dict = semantics.data.get("policy_paths", {})
    semantics_keys = set(policy_paths_dict.keys())
    
    # 校验：whitelist.allowed 的每一项必须在 semantics 中存在。
    missing_in_semantics = allowed_set - semantics_keys
    if missing_in_semantics:
        raise WhitelistSemanticsMismatchError(
            f"policy_path.allowed items missing in policy_path_semantics: {sorted(missing_in_semantics)}"
        )
    
    # 检查闭包一致性。
    consistency_rules = whitelist.data.get("consistency_rules", {})
    if consistency_rules.get("policy_path_must_exist_in_semantics"):
        # 已经检查过 allowed 必须是 semantics 的子集
        pass
    
    # 检查双向一致性。
    extra_in_semantics = semantics_keys - allowed_set
    if extra_in_semantics:
        if consistency_rules.get("policy_path_semantics_must_be_subset_of_whitelist"):
            raise WhitelistSemanticsMismatchError(
                "policy_path_semantics keys not allowed in policy_path.allowed: "
                f"{sorted(extra_in_semantics)}"
            )
        require_exact_cover = consistency_rules.get("require_semantics_exact_cover", False)
        if require_exact_cover:
            raise WhitelistSemanticsMismatchError(
                "policy_path_semantics must exactly cover whitelist policy_path.allowed: "
                f"extra_in_semantics={sorted(extra_in_semantics)}"
            )
        print(
            "[Audit][WARN] policy_path_semantics contains extra entries not in whitelist: "
            f"extra_in_semantics={sorted(extra_in_semantics)}"
        )


def bind_whitelist_to_record(
    record: Dict[str, Any],
    whitelist: RuntimeWhitelist
) -> None:
    """
    功能：将 whitelist 字段写入 record。
    
    Bind whitelist version and digests to record (mutates in-place).
    
    Args:
        record: Record dict to mutate.
        whitelist: Loaded RuntimeWhitelist.
    """
    record["whitelist_version"] = whitelist.whitelist_version
    record["whitelist_digest"] = whitelist.whitelist_digest
    record["whitelist_file_sha256"] = whitelist.whitelist_file_sha256
    record["whitelist_canon_sha256"] = whitelist.whitelist_canon_sha256
    record["whitelist_bound_digest"] = whitelist.whitelist_bound_digest


def bind_semantics_to_record(
    record: Dict[str, Any],
    semantics: PolicyPathSemantics
) -> None:
    """
    功能：将 semantics 字段写入 record。
    
    Bind policy_path_semantics version and digests to record (mutates in-place).
    
    Args:
        record: Record dict to mutate.
        semantics: Loaded PolicyPathSemantics.
    """
    record["policy_path_semantics_version"] = semantics.policy_path_semantics_version
    record["policy_path_semantics_digest"] = semantics.policy_path_semantics_digest
    record["policy_path_semantics_file_sha256"] = semantics.policy_path_semantics_file_sha256
    record["policy_path_semantics_canon_sha256"] = semantics.policy_path_semantics_canon_sha256
    record["policy_path_semantics_bound_digest"] = semantics.policy_path_semantics_bound_digest


def get_pipeline_impl_allowlist(whitelist: RuntimeWhitelist) -> List[str]:
    """
    功能：读取 pipeline_impl_id 允许集合。

    Get pipeline_impl_id allowlist from pipeline registry.

    Args:
        whitelist: Loaded RuntimeWhitelist.

    Returns:
        List of allowed pipeline_impl_id strings.

    Raises:
        TypeError: If whitelist is invalid.
        GateEnforcementError: If allowlist is invalid.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    allowlist = pipeline_registry.list_pipeline_impl_ids()
    if not isinstance(allowlist, list):
        raise GateEnforcementError("pipeline_registry list must be list")
    for value in allowlist:
        if not isinstance(value, str) or not value:
            raise GateEnforcementError("pipeline_registry entries must be non-empty str")
    if len(allowlist) == 0:
        raise GateEnforcementError("pipeline_registry allowlist empty")
    return allowlist


def get_pipeline_realization_policy(whitelist: RuntimeWhitelist) -> Dict[str, Any]:
    """
    功能：读取 pipeline_realization 策略段。

    Pipeline realization policy is disabled (default deny).

    Args:
        whitelist: Loaded RuntimeWhitelist.

    Returns:
        Policy mapping.

    Raises:
        TypeError: If whitelist is invalid.
        GateEnforcementError: If policy is invalid.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    raise GateEnforcementError(
        "pipeline_realization policy disabled",
        gate_name="pipeline_realization.policy.disabled",
        field_path="runtime_whitelist.pipeline_realization",
        expected="<absent>",
        actual="<absent>"
    )


def _raise_pipeline_realization_policy_disabled(field_path: str) -> None:
    """
    功能：拒绝 pipeline_realization 门禁（缺失策略）。

    Reject pipeline_realization checks because policy is disabled.

    Args:
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If field_path is invalid.
        GateEnforcementError: Always raised to enforce default deny.
    """
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    raise GateEnforcementError(
        "pipeline_realization policy disabled",
        gate_name="pipeline_realization.policy.disabled",
        field_path=field_path,
        expected="<absent>",
        actual="<absent>"
    )


def assert_pipeline_model_source_allowed(
    whitelist: RuntimeWhitelist,
    model_source: str,
    field_path: str = "pipeline_provenance.model_source"
) -> None:
    """
    功能：校验 model_source 是否在白名单内。

    Assert model_source is allowed by pipeline_realization policy.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        model_source: Model source string.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If model_source is not allowed.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(model_source, str) or not model_source:
        # model_source 输入不合法，必须 fail-fast。
        raise TypeError("model_source must be non-empty str")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    # 检查 whitelist 中的 pipeline_realization 策略
    pipeline_realization = whitelist.data.get("pipeline_realization")
    if not isinstance(pipeline_realization, dict):
        _raise_pipeline_realization_policy_disabled(field_path)
    
    enabled = pipeline_realization.get("enabled", False)
    if not enabled:
        _raise_pipeline_realization_policy_disabled(field_path)
    
    # 检查 model_source 是否在允许列表中
    allowed_sources = pipeline_realization.get("allowed_model_sources", [])
    if model_source not in allowed_sources:
        raise GateEnforcementError(
            f"model_source not allowed: {model_source}",
            gate_name="pipeline_realization.model_source.not_allowed",
            field_path=field_path,
            expected=f"one of {allowed_sources}",
            actual=model_source
        )


def assert_pipeline_hf_revision_required(
    whitelist: RuntimeWhitelist,
    hf_revision: str,
    field_path: str = "pipeline_provenance.hf_revision"
) -> None:
    """
    功能：校验 hf_revision 必填。

    Assert hf_revision is required by pipeline_realization policy.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        hf_revision: Revision string.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If hf_revision is missing.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    # 检查 whitelist 中的 pipeline_realization 策略
    pipeline_realization = whitelist.data.get("pipeline_realization")
    if not isinstance(pipeline_realization, dict):
        _raise_pipeline_realization_policy_disabled(field_path)
    
    enabled = pipeline_realization.get("enabled", False)
    if not enabled:
        _raise_pipeline_realization_policy_disabled(field_path)
    
    # 检查是否要求 hf_revision
    hf_revision_required = pipeline_realization.get("hf_revision_required", True)
    if hf_revision_required:
        if not isinstance(hf_revision, str) or not hf_revision or hf_revision == "<absent>":
            raise GateEnforcementError(
                "hf_revision required",
                gate_name="pipeline_realization.hf_revision.required",
                field_path=field_path,
                expected="non-empty",
                actual=str(hf_revision)
            )


def assert_pipeline_weights_snapshot_required(
    whitelist: RuntimeWhitelist,
    weights_snapshot_sha256: Any,
    field_path: str = "pipeline_runtime_meta.weights_snapshot_sha256"
) -> None:
    """
    功能：校验 weights_snapshot 必填策略。

    Assert weights_snapshot_sha256 is required by pipeline_realization policy.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        weights_snapshot_sha256: Snapshot digest value.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If weights_snapshot_sha256 is missing when required.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    _raise_pipeline_realization_policy_disabled(field_path)


def assert_pipeline_hf_hub_download_allowed(
    whitelist: RuntimeWhitelist,
    download_allowed: bool,
    field_path: str = "pipeline_provenance.local_files_only"
) -> None:
    """
    功能：校验 HF Hub 下载策略。

    Assert HF Hub download allowed policy matches local_files_only.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        download_allowed: Whether HF download is allowed.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If download policy is violated.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(download_allowed, bool):
        # download_allowed 输入不合法，必须 fail-fast。
        raise TypeError("download_allowed must be bool")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    _raise_pipeline_realization_policy_disabled(field_path)


def assert_inference_device_allowed(
    whitelist: RuntimeWhitelist,
    device: str | None,
    field_path: str = "device"
) -> None:
    """
    功能：校验 inference device 是否在允许集合中。

    Assert that inference device is in allowed list.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        device: Device string or None.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        GateEnforcementError: If device is not allowed.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    _raise_pipeline_realization_policy_disabled(field_path)


def assert_inference_precision_allowed(
    whitelist: RuntimeWhitelist,
    precision: str | None,
    field_path: str = "precision"
) -> None:
    """
    功能：校验 inference precision 是否在允许集合中。

    Assert that inference precision is in allowed list.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        precision: Precision string or None.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        GateEnforcementError: If precision is not allowed.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    _raise_pipeline_realization_policy_disabled(field_path)


def assert_env_fingerprint_required(
    whitelist: RuntimeWhitelist,
    env_fingerprint_canon_sha256: str,
    field_path: str = "env_fingerprint_canon_sha256"
) -> None:
    """
    功能：校验 env_fingerprint 必填。

    Assert env_fingerprint_canon_sha256 is required for real pipeline.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        env_fingerprint_canon_sha256: Canonical digest string.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If env_fingerprint is missing.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    if not isinstance(env_fingerprint_canon_sha256, str) or not env_fingerprint_canon_sha256:
        raise GateEnforcementError(
            "env_fingerprint_canon_sha256 required",
            gate_name="pipeline_realization.env_fingerprint.required",
            field_path=field_path,
            expected="non-empty",
            actual=str(env_fingerprint_canon_sha256)
        )


def assert_pipeline_impl_allowed(
    whitelist: RuntimeWhitelist,
    pipeline_impl_id: str,
    field_path: str = "pipeline_impl_id"
) -> None:
    """
    功能：校验 pipeline_impl_id 是否在白名单内。

    Assert pipeline_impl_id is allowed by runtime whitelist.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        pipeline_impl_id: Pipeline implementation identifier.
        field_path: Field path for error context.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If pipeline_impl_id is not allowed.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")
    if not isinstance(pipeline_impl_id, str) or not pipeline_impl_id:
        # pipeline_impl_id 输入不合法，必须 fail-fast。
        raise TypeError("pipeline_impl_id must be non-empty str")

    allowlist = get_pipeline_impl_allowlist(whitelist)
    if pipeline_impl_id not in allowlist:
        raise GateEnforcementError(
            "pipeline_impl_id not allowed",
            gate_name="pipeline_impl_id.allowlist",
            field_path=field_path,
            expected=str(allowlist),
            actual=pipeline_impl_id
        )


def assert_pipeline_provenance_digest_consistent(record: Dict[str, Any]) -> None:
    """
    功能：校验 pipeline_provenance digest 一致性。

    Assert pipeline_provenance_canon_sha256 matches recomputed digest.

    Args:
        record: Record mapping containing pipeline_provenance fields.

    Returns:
        None.

    Raises:
        TypeError: If record is invalid.
        GateEnforcementError: If digest is missing or mismatched.
    """
    if not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict")

    provenance = record.get("pipeline_provenance")
    if not isinstance(provenance, dict):
        raise GateEnforcementError("pipeline_provenance must be dict")

    digest_value = record.get("pipeline_provenance_canon_sha256")
    if not isinstance(digest_value, str) or not digest_value:
        raise GateEnforcementError("pipeline_provenance_canon_sha256 must be non-empty str")

    try:
        from main.diffusion.sd3.provenance import compute_pipeline_provenance_canon_sha256

        recomputed = compute_pipeline_provenance_canon_sha256(provenance)
    except Exception as exc:
        # digest 计算异常必须显式拒绝。
        raise GateEnforcementError(
            f"pipeline_provenance digest computation failed: {type(exc).__name__}: {exc}"
        ) from exc

    if recomputed != digest_value:
        raise GateEnforcementError(
            "pipeline_provenance digest mismatch",
            gate_name="pipeline_provenance.canon_sha256",
            field_path="pipeline_provenance_canon_sha256",
            expected=recomputed,
            actual=digest_value
        )


def enforce_must_enforce_rules(
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    contracts: FrozenContracts,
    record: Dict[str, Any] | None = None
) -> None:
    """
    功能：执行 runtime_whitelist 中声明的 must_enforce 规则。

    Enforce must_enforce rules declared in runtime_whitelist consistency_rules.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        contracts: Loaded FrozenContracts.
        record: Optional record mapping for record-level checks.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        GateEnforcementError: If a must_enforce rule fails.
    """
    if not isinstance(whitelist, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    if not isinstance(semantics, PolicyPathSemantics):
        # semantics 类型不符合预期，必须 fail-fast。
        raise TypeError("semantics must be PolicyPathSemantics")
    if not isinstance(contracts, FrozenContracts):
        # contracts 类型不符合预期，必须 fail-fast。
        raise TypeError("contracts must be FrozenContracts")
    if record is not None and not isinstance(record, dict):
        # record 类型不符合预期，必须 fail-fast。
        raise TypeError("record must be dict or None")

    rules = whitelist.data.get("consistency_rules", {})
    if not isinstance(rules, dict):
        return

    _override_source_validation(rules, whitelist, record)
    _override_arg_name_validation(rules, whitelist, record)
    _override_arg_name_no_duplicates(rules, whitelist, record)
    _override_applied_audit_required(rules, record)
    _policy_path_closed_under_semantics(rules, whitelist, semantics, record)
    _policy_path_semantics_digest_consistency(rules, semantics, record)
    _policy_path_semantics_fail_reason_enum(rules, semantics, contracts)


def _is_must_enforce(rule_spec: Any) -> bool:
    """
    功能：判定规则是否 must_enforce。

    Check whether a rule spec is marked as must_enforce.

    Args:
        rule_spec: Rule spec mapping or other.

    Returns:
        True if must_enforce; otherwise False.
    """
    return isinstance(rule_spec, dict) and rule_spec.get("enforcement") == "must_enforce"


def _override_source_validation(
    rules: Dict[str, Any],
    whitelist: RuntimeWhitelist,
    record: Dict[str, Any] | None
) -> None:
    """
    功能：验证 override source 枚举与审计一致性。

    Validate override sources against whitelist enums and audit payload.

    Args:
        rules: consistency_rules mapping.
        whitelist: Loaded RuntimeWhitelist.
        record: Optional record mapping.

    Returns:
        None.
    """
    rule_spec = rules.get("override_source_validation")
    if not _is_must_enforce(rule_spec):
        return
    allowed = whitelist.data.get("override", {}).get("source_enum", {}).get("allowed", [])
    if not isinstance(allowed, list):
        raise GateEnforcementError("override.source_enum.allowed must be list")

    _allowed_overrides_sources(whitelist, allowed, "override_source_validation")

    override_applied = record.get("override_applied") if isinstance(record, dict) else None
    if override_applied is None:
        return
    if not isinstance(override_applied, dict):
        raise GateEnforcementError("override_applied must be dict")
    _ensure_override_applied_sources(override_applied, allowed, "override_source_validation")


def _override_arg_name_validation(
    rules: Dict[str, Any],
    whitelist: RuntimeWhitelist,
    record: Dict[str, Any] | None
) -> None:
    """
    功能：验证 override arg_name 枚举与审计一致性。

    Validate override arg_name against whitelist enums and audit payload.

    Args:
        rules: consistency_rules mapping.
        whitelist: Loaded RuntimeWhitelist.
        record: Optional record mapping.

    Returns:
        None.
    """
    rule_spec = rules.get("override_arg_name_validation")
    if not _is_must_enforce(rule_spec):
        return
    allowed = whitelist.data.get("override", {}).get("arg_name_enum", {}).get("allowed", [])
    if not isinstance(allowed, list):
        raise GateEnforcementError("override.arg_name_enum.allowed must be list")

    _allowed_overrides_arg_names(whitelist, allowed, "override_arg_name_validation")

    override_applied = record.get("override_applied") if isinstance(record, dict) else None
    if override_applied is None:
        return
    if not isinstance(override_applied, dict):
        raise GateEnforcementError("override_applied must be dict")
    _ensure_override_applied_arg_names(override_applied, allowed, "override_arg_name_validation")


def _override_arg_name_no_duplicates(
    rules: Dict[str, Any],
    whitelist: RuntimeWhitelist,
    record: Dict[str, Any] | None
) -> None:
    """
    功能：禁止 override arg_name 重复。

    Enforce no duplicate override arg_name in whitelist and audit payloads.

    Args:
        rules: consistency_rules mapping.
        whitelist: Loaded RuntimeWhitelist.
        record: Optional record mapping.

    Returns:
        None.
    """
    rule_spec = rules.get("override_arg_name_no_duplicates")
    if not _is_must_enforce(rule_spec):
        return

    allowed_overrides = whitelist.data.get("override", {}).get("allowed_overrides", [])
    if not isinstance(allowed_overrides, list):
        raise GateEnforcementError("override.allowed_overrides must be list")

    seen = set()
    for entry in allowed_overrides:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override.allowed_overrides entries must be dict")
        arg_name = entry.get("arg_name")
        if not isinstance(arg_name, str) or not arg_name:
            raise GateEnforcementError("override.allowed_overrides arg_name must be non-empty str")
        if arg_name in seen:
            raise GateEnforcementError(f"override_arg_name_duplicated: arg_name={arg_name}")
        seen.add(arg_name)

    override_applied = record.get("override_applied") if isinstance(record, dict) else None
    if override_applied is None:
        return
    if not isinstance(override_applied, dict):
        raise GateEnforcementError("override_applied must be dict")

    applied = override_applied.get("applied_fields", [])
    if not isinstance(applied, list):
        raise GateEnforcementError("override_applied.applied_fields must be list")
    applied_seen = set()
    for entry in applied:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override_applied.applied_fields entries must be dict")
        arg_name = entry.get("arg_name")
        if not isinstance(arg_name, str) or not arg_name:
            raise GateEnforcementError("override_applied.applied_fields arg_name must be non-empty str")
        if arg_name in applied_seen:
            raise GateEnforcementError(f"override_arg_name_duplicated: arg_name={arg_name}")
        applied_seen.add(arg_name)


def _override_applied_audit_required(
    rules: Dict[str, Any],
    record: Dict[str, Any] | None
) -> None:
    """
    功能：override_applied 审计段最小字段校验。

    Validate override_applied audit payload contains minimum required fields.

    Args:
        rules: consistency_rules mapping.
        record: Optional record mapping.

    Returns:
        None.
    """
    rule_spec = rules.get("override_applied_audit_required")
    if not _is_must_enforce(rule_spec):
        return
    if record is None:
        return
    override_applied = record.get("override_applied")
    if override_applied is None:
        return
    if not isinstance(override_applied, dict):
        raise GateEnforcementError("override_applied must be dict")
    minimum_fields = rule_spec.get("override_applied_minimum_fields", [])
    if not isinstance(minimum_fields, list):
        raise GateEnforcementError("override_applied_minimum_fields must be list")
    applied = override_applied.get("applied_fields", [])
    if not isinstance(applied, list):
        raise GateEnforcementError("override_applied.applied_fields must be list")
    for entry in applied:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override_applied.applied_fields entries must be dict")
        for field_name in minimum_fields:
            value = entry.get(field_name)
            if value is None:
                raise GateEnforcementError(
                    f"override_applied missing field: field_name={field_name}"
                )
            if isinstance(value, str) and not value:
                raise GateEnforcementError(
                    f"override_applied empty field: field_name={field_name}"
                )


def _policy_path_closed_under_semantics(
    rules: Dict[str, Any],
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    record: Dict[str, Any] | None
) -> None:
    """
    功能：policy_path 与语义闭包校验。

    Validate policy_path is in whitelist.allowed and exists in semantics.

    Args:
        rules: consistency_rules mapping.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        record: Optional record mapping.

    Returns:
        None.
    """
    rule_spec = rules.get("policy_path_closed_under_semantics")
    if not _is_must_enforce(rule_spec):
        return
    if record is None:
        return
    policy_path = record.get("policy_path")
    if not isinstance(policy_path, str) or not policy_path:
        raise GateEnforcementError("policy_path_not_closed_under_semantics: missing policy_path")
    allowed = whitelist.data.get("policy_path", {}).get("allowed", [])
    if not isinstance(allowed, list):
        raise GateEnforcementError("policy_path.allowed must be list")
    if policy_path not in allowed:
        raise GateEnforcementError(
            f"policy_path_not_closed_under_semantics: value={policy_path}"
        )
    semantics_paths = semantics.data.get("policy_paths", {})
    if not isinstance(semantics_paths, dict):
        raise GateEnforcementError("policy_path_semantics.policy_paths must be dict")
    if policy_path not in semantics_paths:
        raise GateEnforcementError(
            f"policy_path_not_closed_under_semantics: value={policy_path}"
        )


def _policy_path_semantics_digest_consistency(
    rules: Dict[str, Any],
    semantics: PolicyPathSemantics,
    record: Dict[str, Any] | None
) -> None:
    """
    功能：policy_path_semantics_digest 一致性校验。

    Validate record policy_path_semantics_digest matches loaded semantics.

    Args:
        rules: consistency_rules mapping.
        semantics: Loaded PolicyPathSemantics.
        record: Optional record mapping.

    Returns:
        None.
    """
    rule_spec = rules.get("policy_path_semantics_digest_consistency")
    if not _is_must_enforce(rule_spec):
        return
    if record is None:
        return
    record_digest = record.get("policy_path_semantics_digest")
    if not isinstance(record_digest, str) or not record_digest:
        raise GateEnforcementError("semantics_digest_mismatch: missing policy_path_semantics_digest")
    if record_digest != semantics.policy_path_semantics_digest:
        raise GateEnforcementError(
            "semantics_digest_mismatch: "
            f"record={record_digest}, expected={semantics.policy_path_semantics_digest}"
        )


def _policy_path_semantics_fail_reason_enum(
    rules: Dict[str, Any],
    semantics: PolicyPathSemantics,
    contracts: FrozenContracts
) -> None:
    """
    功能：policy_path_semantics record_fail_reason 枚举闭包校验。

    Validate policy_path_semantics record_fail_reason values are in frozen enum.

    Args:
        rules: consistency_rules mapping.
        semantics: Loaded PolicyPathSemantics.
        contracts: Loaded FrozenContracts.

    Returns:
        None.
    """
    rule_spec = rules.get("policy_path_semantics_record_fail_reason_in_frozen_fail_reason_enum")
    if not _is_must_enforce(rule_spec):
        return
    interpretation = get_contract_interpretation(contracts)
    allowed = interpretation.fail_reason_enum.allowed_values
    if not isinstance(allowed, list):
        raise GateEnforcementError("fail_reason_enum.allowed_values must be list")
    allowed_set = set(allowed)

    policy_paths = semantics.data.get("policy_paths", {})
    if not isinstance(policy_paths, dict):
        raise GateEnforcementError("policy_path_semantics.policy_paths must be dict")
    for policy_name, policy_value in policy_paths.items():
        if not isinstance(policy_value, dict):
            continue
        on_chain_failure = policy_value.get("on_chain_failure", {})
        if not isinstance(on_chain_failure, dict):
            continue
        for chain_name, chain_spec in on_chain_failure.items():
            if not isinstance(chain_spec, dict):
                continue
            record_fail_reason = chain_spec.get("record_fail_reason")
            if record_fail_reason is None:
                continue
            if not isinstance(record_fail_reason, str) or not record_fail_reason:
                raise GateEnforcementError(
                    "policy_path_semantics record_fail_reason must be non-empty str: "
                    f"policy_path={policy_name}, chain={chain_name}"
                )
            if record_fail_reason not in allowed_set:
                raise GateEnforcementError(
                    "policy_path_semantics record_fail_reason not allowed: "
                    f"policy_path={policy_name}, chain={chain_name}, "
                    f"value={record_fail_reason}"
                )


def _allowed_overrides_sources(
    whitelist: RuntimeWhitelist,
    allowed_sources: List[str],
    rule_name: str
) -> None:
    """
    功能：校验 whitelist.allowed_overrides 的 source 枚举。

    Validate whitelist allowed_overrides entries against source enum.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        allowed_sources: Allowed source list.
        rule_name: Rule name for context.

    Returns:
        None.
    """
    allowed_overrides = whitelist.data.get("override", {}).get("allowed_overrides", [])
    if not isinstance(allowed_overrides, list):
        raise GateEnforcementError("override.allowed_overrides must be list")
    for entry in allowed_overrides:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override.allowed_overrides entries must be dict")
        source = entry.get("source")
        if source not in allowed_sources:
            raise GateEnforcementError(
                f"{rule_name}: source not allowed: source={source}"
            )


def _allowed_overrides_arg_names(
    whitelist: RuntimeWhitelist,
    allowed_arg_names: List[str],
    rule_name: str
) -> None:
    """
    功能：校验 whitelist.allowed_overrides 的 arg_name 枚举。

    Validate whitelist allowed_overrides entries against arg_name enum.

    Args:
        whitelist: Loaded RuntimeWhitelist.
        allowed_arg_names: Allowed arg_name list.
        rule_name: Rule name for context.

    Returns:
        None.
    """
    allowed_overrides = whitelist.data.get("override", {}).get("allowed_overrides", [])
    if not isinstance(allowed_overrides, list):
        raise GateEnforcementError("override.allowed_overrides must be list")
    for entry in allowed_overrides:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override.allowed_overrides entries must be dict")
        arg_name = entry.get("arg_name")
        if arg_name not in allowed_arg_names:
            raise GateEnforcementError(
                f"{rule_name}: arg_name not allowed: arg_name={arg_name}"
            )


def _ensure_override_applied_sources(
    override_applied: Dict[str, Any],
    allowed_sources: List[str],
    rule_name: str
) -> None:
    """
    功能：校验 override_applied 中的 source 枚举。

    Validate override_applied sources against allowed list.

    Args:
        override_applied: override_applied mapping.
        allowed_sources: Allowed source list.
        rule_name: Rule name for context.

    Returns:
        None.
    """
    source = override_applied.get("source")
    if source not in allowed_sources:
        raise GateEnforcementError(
            f"{rule_name}: override_applied source not allowed: source={source}"
        )
    applied = override_applied.get("applied_fields", [])
    if not isinstance(applied, list):
        raise GateEnforcementError("override_applied.applied_fields must be list")
    for entry in applied:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override_applied.applied_fields entries must be dict")
        entry_source = entry.get("source")
        if entry_source not in allowed_sources:
            raise GateEnforcementError(
                f"{rule_name}: override_applied source not allowed: source={entry_source}"
            )


def _ensure_override_applied_arg_names(
    override_applied: Dict[str, Any],
    allowed_arg_names: List[str],
    rule_name: str
) -> None:
    """
    功能：校验 override_applied 中的 arg_name 枚举。

    Validate override_applied arg_name values against allowed list.

    Args:
        override_applied: override_applied mapping.
        allowed_arg_names: Allowed arg_name list.
        rule_name: Rule name for context.

    Returns:
        None.
    """
    applied = override_applied.get("applied_fields", [])
    if not isinstance(applied, list):
        raise GateEnforcementError("override_applied.applied_fields must be list")
    for entry in applied:
        if not isinstance(entry, dict):
            raise GateEnforcementError("override_applied.applied_fields entries must be dict")
        arg_name = entry.get("arg_name")
        if arg_name not in allowed_arg_names:
            raise GateEnforcementError(
                f"{rule_name}: override_applied arg_name not allowed: arg_name={arg_name}"
            )


def assert_impl_allowed(
    whitelist: RuntimeWhitelist,
    impl_identity: Dict[str, str]
) -> None:
    """
    功能：校验 impl_identity 中的 impl_id 是否被白名单允许。
    
    Validate that all impl_id values in impl_identity are in whitelist's allowed list.
    Fail-fast if any impl_id is not allowed.
    
    Args:
        whitelist: Loaded RuntimeWhitelist.
        impl_identity: Dict with keys: content_extractor_id, geometry_extractor_id, 
                      fusion_rule_id, subspace_planner_id, sync_module_id.
    
    Returns:
        None.
    
    Raises:
        GateEnforcementError: If any impl_id is not allowed by whitelist.
        TypeError: If whitelist or impl_identity are invalid.
        ValueError: If impl_identity is missing required fields.
    """
    # 输入类型检查，确保 whitelist 和 impl_identity 符合预期结构。
    if not hasattr(whitelist, 'data'):
        raise TypeError(f"whitelist must have .data attribute, got {type(whitelist)}")
    
    if not isinstance(impl_identity, dict):
        raise TypeError(f"impl_identity must be dict, got {type(impl_identity)}")
    
    # 定义域与对应的 impl_id 字段名匹配。
    domain_field_mapping = {
        "content_extractor": "content_extractor_id",
        "geometry_extractor": "geometry_extractor_id",
        "fusion_rule": "fusion_rule_id",
        "subspace_planner": "subspace_planner_id",
        "sync_module": "sync_module_id",
    }
    
    # 从 whitelist 中提取 impl_id 白名单。
    impl_cfg = whitelist.data.get("impl_id")
    legacy_impl_cfg = whitelist.data.get("impl")
    if impl_cfg is None:
        # whitelist 缺失 impl_id，必须 fail-fast。
        if legacy_impl_cfg is not None:
            raise GateEnforcementError("whitelist.impl_id missing while legacy impl is present")
        raise GateEnforcementError("whitelist.impl_id missing")
    if not isinstance(impl_cfg, dict):
        raise ValueError("whitelist.impl_id must be dict")
    if legacy_impl_cfg is not None:
        if not isinstance(legacy_impl_cfg, dict):
            raise ValueError("whitelist.impl must be dict when provided")
        legacy_allowed_by_domain = legacy_impl_cfg.get("allowed_by_domain")
        legacy_allow_unknown_domain = legacy_impl_cfg.get("allow_unknown_domain", False)
        if legacy_allowed_by_domain != impl_cfg.get("allowed_by_domain") or \
            legacy_allow_unknown_domain != impl_cfg.get("allow_unknown_domain", False):
            raise GateEnforcementError("whitelist.impl and whitelist.impl_id conflict")
    
    allowed_by_domain = impl_cfg.get("allowed_by_domain", {})
    if not isinstance(allowed_by_domain, dict):
        raise ValueError("whitelist.impl_id.allowed_by_domain must be dict")
    
    allow_unknown_domain = impl_cfg.get("allow_unknown_domain", False)
    
    # 逐个检查 impl_identity 中的 impl_id。
    for domain, field_name in domain_field_mapping.items():
        impl_id = impl_identity.get(field_name)
        
        if not isinstance(impl_id, str) or not impl_id:
            raise ValueError(f"impl_identity[{field_name}] must be non-empty str, got {impl_id}")
        
        # 获取该域的允许列表。
        allowed_list = allowed_by_domain.get(domain)
        
        if allowed_list is None:
            # domain 不在 whitelist 中。
            if not allow_unknown_domain:
                raise GateEnforcementError(
                    f"Domain '{domain}' not found in whitelist.impl_id.allowed_by_domain, "
                    f"and allow_unknown_domain=false"
                )
            # 否则允许（if allow_unknown_domain=true）
            continue
        
        if not isinstance(allowed_list, list):
            raise ValueError(f"whitelist.impl_id.allowed_by_domain[{domain}] must be list")
        
        # 检查 impl_id 是否在允许列表中
        if impl_id not in allowed_list:
            raise GateEnforcementError(
                f"impl_id '{impl_id}' for domain '{domain}' not allowed by whitelist. "
                f"Allowed: {allowed_list}"
            )


def bind_impl_identity_to_record(
    record: Dict[str, Any],
    impl_identity: Dict[str, str],
    impl_identity_digest: str
) -> None:
    """
    功能：将 impl_identity 与摘要写入 record。
    
    Bind impl_identity and impl_identity_digest to record (mutates in-place).
    
    Args:
        record: Record dict to mutate.
        impl_identity: ImplIdentity as dict.
        impl_identity_digest: SHA256 digest of impl_identity.
    
    Returns:
        None.
    
    Raises:
        TypeError: If record is not dict.
    """
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record)}")
    
    record["impl_identity"] = impl_identity
    record["impl_identity_digest"] = impl_identity_digest


def bind_impl_meta_to_record(
    record: Dict[str, Any],
    impl_meta: Dict[str, Any],
    impl_meta_digest: str
) -> None:
    """
    功能：将 impl_meta 与摘要写入 record。
    
    Bind impl_meta and impl_meta_digest to record (mutates in-place).
    
    Args:
        record: Record dict to mutate.
        impl_meta: ImplMeta as dict.
        impl_meta_digest: SHA256 digest of impl_meta.
    
    Returns:
        None.
    
    Raises:
        TypeError: If record is not dict.
    """
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record)}")
    
    record["impl_meta"] = impl_meta
    record["impl_meta_digest"] = impl_meta_digest
