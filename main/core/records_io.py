"""
records 写盘唯一入口

功能说明：
- 所有 records 和相关输出的写盘必须通过本模块进行，并触发 freeze_gate 校验。
- frozen write entry point declaration: 本模块是 records 的 unique write entry point。
  - 任何绕过（bypass）该入口的写盘行为均属违规并可被审计扫描。
- 实现原子写盘以保证稳定性：在同目录下创建临时文件，fsync，最后替换目标文件。
- 写盘通道架构（frozen）：
    (1) Records 写盘（records_dir）：
        - write_json：JSON 格式覆盖写入，触发门禁校验。
        - append_jsonl：JSONL 格式追加写入，触发门禁校验。
        - 两者都强制要求：必须是 dict，必须通过 freeze_gate.assert_prewrite() 校验。
    (2) Artifact 写盘（artifacts_dir）：
        - write_artifact_json：标准 artifact 输出（通过 fact sources 进行门禁校验）。
        - write_artifact_json_unbound：早期失败的兜底 artifact 输出（最小化检查）。
        - 两者都强制要求：必须是 dict，必须不越界 artifacts_dir。
    (3) Log 写盘（logs_dir）：
        - write_text：文本内容输出到 logs。
        - 强制要求：必须不越界 logs_dir。
- 所有写盘操作都必须是原子性的（临时文件 + fsync + 替换）。
- 不允许存在除这五个入口之外的隐藏写盘通道。
- 执行门禁校验阻止任何不合规的记录写入。
"""

import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, cast

from main.core import schema
from main.core.contracts import FrozenContracts, ContractInterpretation, get_contract_interpretation
from main.core.injection_scope import InjectionScopeManifest
from main.core.errors import FactSourcesNotInitializedError, RecordsWritePolicyError
from main.policy.freeze_gate import assert_prewrite
from main.policy.runtime_whitelist import RuntimeWhitelist, PolicyPathSemantics
from main.policy import path_policy


@dataclass(frozen=True)
class FactSourcesContext:
    """
    功能：写盘事实源上下文。

    Write-time fact sources context for gate enforcement and path isolation.

    Args:
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        injection_scope_manifest: Loaded InjectionScopeManifest.
        run_root: Run root directory.
        records_dir: Records output directory.
        artifacts_dir: Artifacts output directory.
        logs_dir: Logs output directory.
    """

    contracts: FrozenContracts
    whitelist: RuntimeWhitelist
    semantics: PolicyPathSemantics
    injection_scope_manifest: InjectionScopeManifest
    run_root: Path
    records_dir: Path
    artifacts_dir: Path
    logs_dir: Path


_FACT_SOURCES_CONTEXT: ContextVar[Optional[FactSourcesContext]] = ContextVar(
    "fact_sources_context",
    default=None
)

_RECOMMENDED_ENFORCE_REPORT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "recommended_enforce_report",
    default=None
)

_ARTIFACT_ANCHOR_ALLOWLIST = {
    "run_closure.json",
    "records_manifest.json",
    "cfg_audit/cfg_audit.json",
    "signoff/signoff_report.json",
    "signoff/signoff_bundle/signoff_report_canon.json"
}

_ARTIFACT_CFG_AUDIT_ALLOWED_FIELDS = {
    "_artifact_audit",
    "config_path",
    "overrides_applied_summary",
    "cfg_pruned_for_digest_canon_sha256",
    "cfg_digest",
    "cfg_audit_canon_sha256"
}

_ARTIFACT_FORBIDDEN_ANCHOR_FIELDS = {
    "contract_bound_digest",
    "whitelist_bound_digest",
    "policy_path_semantics_bound_digest",
    "injection_scope_manifest_bound_digest"
}


def _resolve_governed_artifact_contract(
    contracts: FrozenContracts,
    relative_path: str,
) -> Dict[str, Any] | None:
    """
    功能：解析受治理的 artifact 契约条目。

    Resolve the governed artifact contract entry for a relative artifact path.

    Args:
        contracts: Loaded frozen contracts.
        relative_path: Artifact path relative to artifacts_dir.

    Returns:
        Matching artifact contract mapping, or None when path is not governed.

    Raises:
        RecordsWritePolicyError: If the artifact governance config is malformed.
    """
    if not isinstance(contracts, FrozenContracts):
        # contracts 类型不符合预期，必须 fail-fast。
        raise TypeError("contracts must be FrozenContracts")
    if not isinstance(relative_path, str) or not relative_path:
        # relative_path 输入不合法，必须 fail-fast。
        raise ValueError("relative_path must be non-empty str")

    artifact_schema = contracts.data.get("artifact_schema")
    if artifact_schema is None:
        return None
    if not isinstance(artifact_schema, dict):
        # artifact_schema 结构损坏会破坏正式治理，必须 fail-fast。
        raise RecordsWritePolicyError("frozen_contracts artifact_schema must be mapping")

    append_only = artifact_schema.get("append_only")
    if append_only is not True:
        # artifact_schema 必须显式声明 append_only=true。
        raise RecordsWritePolicyError("frozen_contracts artifact_schema must declare append_only=true")

    artifact_contracts = artifact_schema.get("artifact_contracts")
    if not isinstance(artifact_contracts, dict):
        # artifact_contracts 缺失会导致治理裸奔，必须 fail-fast。
        raise RecordsWritePolicyError("frozen_contracts artifact_schema.artifact_contracts must be mapping")

    for artifact_type, spec in artifact_contracts.items():
        if not isinstance(artifact_type, str) or not artifact_type:
            # artifact_type 名称非法，必须 fail-fast。
            raise RecordsWritePolicyError("artifact contract key must be non-empty str")
        if not isinstance(spec, dict):
            # 条目类型损坏，必须 fail-fast。
            raise RecordsWritePolicyError(
                f"artifact contract must be mapping: artifact_type={artifact_type}"
            )
        spec_relative_path = spec.get("relative_path")
        if not isinstance(spec_relative_path, str) or not spec_relative_path:
            # 相对路径缺失会破坏路径治理，必须 fail-fast。
            raise RecordsWritePolicyError(
                f"artifact contract relative_path must be non-empty str: artifact_type={artifact_type}"
            )
        if spec_relative_path != relative_path:
            continue

        allowed_top_level_fields = spec.get("allowed_top_level_fields")
        if not isinstance(allowed_top_level_fields, list) or not allowed_top_level_fields:
            # 白名单集合缺失会破坏字段治理，必须 fail-fast。
            raise RecordsWritePolicyError(
                f"artifact contract allowed_top_level_fields must be non-empty list: artifact_type={artifact_type}"
            )
        if not all(isinstance(field_name, str) and field_name for field_name in allowed_top_level_fields):
            # 白名单成员必须是非空字符串。
            raise RecordsWritePolicyError(
                f"artifact contract allowed_top_level_fields must contain only non-empty str: artifact_type={artifact_type}"
            )

        return {
            "artifact_type": artifact_type,
            "relative_path": spec_relative_path,
            "allowed_top_level_fields": [str(field_name) for field_name in allowed_top_level_fields],
        }
    return None


def _enforce_governed_artifact_contract(
    obj: Dict[str, Any],
    dst_path: Path,
    relative_path: str,
    contracts: FrozenContracts,
) -> None:
    """
    功能：对受治理 artifact 执行精确顶层字段白名单校验。

    Enforce exact top-level allowlist validation for governed artifacts.

    Args:
        obj: Artifact payload mapping.
        dst_path: Destination artifact path.
        relative_path: Artifact path relative to artifacts_dir.
        contracts: Loaded frozen contracts.

    Returns:
        None.

    Raises:
        RecordsWritePolicyError: If artifact payload drifts from the governed contract.
    """
    governed_contract = _resolve_governed_artifact_contract(contracts, relative_path)
    if governed_contract is None:
        return

    artifact_type = obj.get("artifact_type")
    expected_artifact_type = governed_contract["artifact_type"]
    if artifact_type != expected_artifact_type:
        # artifact_type 与治理契约不一致，必须 fail-fast。
        raise RecordsWritePolicyError(
            "artifact payload type does not match governed contract: "
            f"path={dst_path}, expected={expected_artifact_type}, actual={artifact_type}"
        )

    top_level_fields = _collect_top_level_keys(obj)
    expected_fields = set(cast(List[str], governed_contract["allowed_top_level_fields"]))
    extra_top_level = sorted(top_level_fields - expected_fields - {"_artifact_audit"})
    missing_top_level = sorted(expected_fields - top_level_fields)
    if extra_top_level or missing_top_level:
        # 受治理 artifact 必须严格满足 append-only 白名单闭包。
        raise RecordsWritePolicyError(
            "artifact payload violates governed top-level allowlist: "
            f"path={dst_path}, missing={missing_top_level}, extra={extra_top_level}"
        )


@contextmanager
def bound_fact_sources(
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    run_root: Path,
    records_dir: Path,
    artifacts_dir: Path,
    logs_dir: Path,
    *,
    injection_scope_manifest: InjectionScopeManifest
) -> Iterator[None]:
    """
    功能：绑定事实源写盘上下文。

    Bind fact sources and run layout for controlled write operations.

    Args:
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        injection_scope_manifest: Loaded InjectionScopeManifest.
        run_root: Run root directory.
        records_dir: Records output directory.
        artifacts_dir: Artifacts output directory.
        logs_dir: Logs output directory.

    Yields:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If layout validation fails.
    """
    contracts_obj: Any = contracts
    if not isinstance(contracts_obj, FrozenContracts):
        # contracts 类型不符合预期，必须 fail-fast。
        raise TypeError("contracts must be FrozenContracts")
    whitelist_obj: Any = whitelist
    if not isinstance(whitelist_obj, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    semantics_obj: Any = semantics
    if not isinstance(semantics_obj, PolicyPathSemantics):
        # semantics 类型不符合预期，必须 fail-fast。
        raise TypeError("semantics must be PolicyPathSemantics")
    injection_scope_manifest_obj: Any = injection_scope_manifest
    if not isinstance(injection_scope_manifest_obj, InjectionScopeManifest):
        # injection_scope_manifest 类型不符合预期，必须 fail-fast。
        raise TypeError("injection_scope_manifest must be InjectionScopeManifest")
    run_root_obj: Any = run_root
    if not isinstance(run_root_obj, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    records_dir_obj: Any = records_dir
    if not isinstance(records_dir_obj, Path):
        # records_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("records_dir must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    logs_dir_obj: Any = logs_dir
    if not isinstance(logs_dir_obj, Path):
        # logs_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("logs_dir must be Path")

    normalized_contracts = contracts_obj
    normalized_whitelist = whitelist_obj
    normalized_semantics = semantics_obj
    normalized_injection_scope_manifest = injection_scope_manifest_obj
    run_root = run_root_obj.resolve()
    records_dir = records_dir_obj.resolve()
    artifacts_dir = artifacts_dir_obj.resolve()
    logs_dir = logs_dir_obj.resolve()

    path_policy.validate_output_target(records_dir, "record", run_root)
    path_policy.validate_output_target(artifacts_dir, "artifact", run_root)
    path_policy.validate_output_target(logs_dir, "log", run_root)

    ctx = FactSourcesContext(
        contracts=normalized_contracts,
        whitelist=normalized_whitelist,
        semantics=normalized_semantics,
        injection_scope_manifest=normalized_injection_scope_manifest,
        run_root=run_root,
        records_dir=records_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir
    )
    token = _FACT_SOURCES_CONTEXT.set(ctx)
    report_token = _RECOMMENDED_ENFORCE_REPORT.set(None)
    try:
        yield
    finally:
        _RECOMMENDED_ENFORCE_REPORT.reset(report_token)
        _FACT_SOURCES_CONTEXT.reset(token)


def get_bound_fact_sources() -> Dict[str, Any]:
    """
    功能：获取已绑定的事实源摘要。

    Get bound fact sources metadata from the active context.

    Args:
        None.

    Returns:
        Dict with contract/whitelist/policy_path_semantics version and digests.

    Raises:
        FactSourcesNotInitializedError: If fact sources are not initialized.
    """
    ctx = _require_fact_sources_initialized()
    return {
        "contract_version": ctx.contracts.contract_version,
        "contract_digest": ctx.contracts.contract_digest,
        "contract_file_sha256": ctx.contracts.contract_file_sha256,
        "contract_canon_sha256": ctx.contracts.contract_canon_sha256,
        "contract_bound_digest": ctx.contracts.contract_bound_digest,
        "whitelist_version": ctx.whitelist.whitelist_version,
        "whitelist_digest": ctx.whitelist.whitelist_digest,
        "whitelist_file_sha256": ctx.whitelist.whitelist_file_sha256,
        "whitelist_canon_sha256": ctx.whitelist.whitelist_canon_sha256,
        "whitelist_bound_digest": ctx.whitelist.whitelist_bound_digest,
        "policy_path_semantics_version": ctx.semantics.policy_path_semantics_version,
        "policy_path_semantics_digest": ctx.semantics.policy_path_semantics_digest,
        "policy_path_semantics_file_sha256": ctx.semantics.policy_path_semantics_file_sha256,
        "policy_path_semantics_canon_sha256": ctx.semantics.policy_path_semantics_canon_sha256,
        "policy_path_semantics_bound_digest": ctx.semantics.policy_path_semantics_bound_digest,
        "injection_scope_manifest_version": ctx.injection_scope_manifest.injection_scope_manifest_version,
        "injection_scope_manifest_digest": ctx.injection_scope_manifest.injection_scope_manifest_digest,
        "injection_scope_manifest_file_sha256": ctx.injection_scope_manifest.injection_scope_manifest_file_sha256,
        "injection_scope_manifest_canon_sha256": ctx.injection_scope_manifest.injection_scope_manifest_canon_sha256,
        "injection_scope_manifest_bound_digest": ctx.injection_scope_manifest.injection_scope_manifest_bound_digest
    }


def build_fact_sources_snapshot(
    contracts: FrozenContracts,
    whitelist: RuntimeWhitelist,
    semantics: PolicyPathSemantics,
    injection_scope_manifest: InjectionScopeManifest
) -> Dict[str, Any]:
    """
    功能：构建事实源快照。

    Build fact sources snapshot without ContextVar or disk writes.

    Args:
        contracts: Loaded FrozenContracts.
        whitelist: Loaded RuntimeWhitelist.
        semantics: Loaded PolicyPathSemantics.
        injection_scope_manifest: Loaded InjectionScopeManifest.

    Returns:
        Dict with contract/whitelist/policy_path_semantics version and digests.

    Raises:
        TypeError: If inputs are invalid.
    """
    contracts_obj: Any = contracts
    if not isinstance(contracts_obj, FrozenContracts):
        # contracts 类型不符合预期，必须 fail-fast。
        raise TypeError("contracts must be FrozenContracts")
    whitelist_obj: Any = whitelist
    if not isinstance(whitelist_obj, RuntimeWhitelist):
        # whitelist 类型不符合预期，必须 fail-fast。
        raise TypeError("whitelist must be RuntimeWhitelist")
    semantics_obj: Any = semantics
    if not isinstance(semantics_obj, PolicyPathSemantics):
        # semantics 类型不符合预期，必须 fail-fast。
        raise TypeError("semantics must be PolicyPathSemantics")
    injection_scope_manifest_obj: Any = injection_scope_manifest
    if not isinstance(injection_scope_manifest_obj, InjectionScopeManifest):
        # injection_scope_manifest 类型不符合预期，必须 fail-fast。
        raise TypeError("injection_scope_manifest must be InjectionScopeManifest")

    normalized_contracts = contracts_obj
    normalized_whitelist = whitelist_obj
    normalized_semantics = semantics_obj
    normalized_injection_scope_manifest = injection_scope_manifest_obj

    return {
        "contract_version": normalized_contracts.contract_version,
        "contract_digest": normalized_contracts.contract_digest,
        "contract_file_sha256": normalized_contracts.contract_file_sha256,
        "contract_canon_sha256": normalized_contracts.contract_canon_sha256,
        "contract_bound_digest": normalized_contracts.contract_bound_digest,
        "whitelist_version": normalized_whitelist.whitelist_version,
        "whitelist_digest": normalized_whitelist.whitelist_digest,
        "whitelist_file_sha256": normalized_whitelist.whitelist_file_sha256,
        "whitelist_canon_sha256": normalized_whitelist.whitelist_canon_sha256,
        "whitelist_bound_digest": normalized_whitelist.whitelist_bound_digest,
        "policy_path_semantics_version": normalized_semantics.policy_path_semantics_version,
        "policy_path_semantics_digest": normalized_semantics.policy_path_semantics_digest,
        "policy_path_semantics_file_sha256": normalized_semantics.policy_path_semantics_file_sha256,
        "policy_path_semantics_canon_sha256": normalized_semantics.policy_path_semantics_canon_sha256,
        "policy_path_semantics_bound_digest": normalized_semantics.policy_path_semantics_bound_digest,
        "injection_scope_manifest_version": normalized_injection_scope_manifest.injection_scope_manifest_version,
        "injection_scope_manifest_digest": normalized_injection_scope_manifest.injection_scope_manifest_digest,
        "injection_scope_manifest_file_sha256": normalized_injection_scope_manifest.injection_scope_manifest_file_sha256,
        "injection_scope_manifest_canon_sha256": normalized_injection_scope_manifest.injection_scope_manifest_canon_sha256,
        "injection_scope_manifest_bound_digest": normalized_injection_scope_manifest.injection_scope_manifest_bound_digest
    }


def get_recommended_enforce_report() -> Optional[Dict[str, Any]]:
    """
    功能：获取推荐门禁审计报告。

    Get the latest recommended_enforce audit report if available.

    Args:
        None.

    Returns:
        Audit report dict or None if not available.

    Raises:
        FactSourcesNotInitializedError: If fact sources are not initialized.
    """
    _require_fact_sources_initialized()
    report = _RECOMMENDED_ENFORCE_REPORT.get()
    if report is None:
        return None
    report_obj: Any = report
    if not isinstance(report_obj, dict):
        # 审计报告类型不合法，必须 fail-fast。
        raise TypeError("recommended_enforce_report must be dict")
    return dict(cast(Dict[str, Any], report_obj))


def _update_recommended_enforce_report(report: Optional[Dict[str, Any]]) -> None:
    """
    功能：更新推荐门禁审计报告缓存。

    Update cached recommended_enforce audit report.

    Args:
        report: Audit report dict from freeze_gate.assert_prewrite.

    Returns:
        None.

    Raises:
        TypeError: If report is not dict when provided.
    """
    if report is None:
        return
    report_obj: Any = report
    if not isinstance(report_obj, dict):
        # report 类型不合法，必须 fail-fast。
        raise TypeError("recommended_enforce_report must be dict")
    _RECOMMENDED_ENFORCE_REPORT.set(dict(cast(Dict[str, Any], report_obj)))


def _ensure_parent_dir(dst: Path) -> None:
    """
    功能：创建目标文件的父目录。
    
    Create parent directories for destination path.
    
    Args:
        dst: Destination file path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)


def _is_windows_winerror_5(exc: BaseException) -> bool:
    """
    功能：判断是否为 Windows 下 replace 触发的 WinError 5。 

    Detect whether an exception is Windows access-denied (WinError 5).

    Args:
        exc: Exception instance raised during replace.

    Returns:
        True only when running on Windows and the exception is PermissionError with winerror=5.
    """
    if os.name != "nt":
        return False
    if not isinstance(exc, PermissionError):
        return False
    return getattr(exc, "winerror", None) == 5


def _json_dumps_stable(
    obj: Any,
    indent: Optional[int],
    ensure_ascii: bool,
    *,
    compact: bool = False
) -> str:
    """
    功能：使用稳定 sort_keys 的 JSON 序列化。
    
    Serialize object to JSON string with stable key ordering.
    
    Args:
        obj: Object to serialize.
        indent: Indent level (None for compact output).
        ensure_ascii: Whether to ensure ASCII encoding.
        compact: If True, use compact separators (":", ","); else default.
    
    Returns:
        JSON string with sort_keys=True for stable output.
    """
    if compact:
        # jsonl 格式。
        return json.dumps(
            obj,
            indent=None,
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
            sort_keys=True
        )
    else:
        # write_json 格式。
        return json.dumps(
            obj,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=True
        )


def write_path_validation_audit(
    audit_record: Dict[str, Any],
    audit_records_dir: Optional[Path] = None
) -> Path:
    """
    功能：写入路径校验审计记录。
    
    Write a path validation audit record to artifacts directory.
    
    Args:
        audit_record: Audit record from path_policy.build_path_validation_audit_record().
        audit_records_dir: Optional directory to write audit records. If None, uses artifacts_dir.
    
    Returns:
        Path to written audit record file.
    
    Raises:
        TypeError: If audit_record is not dict.
        FactSourcesNotInitializedError: If fact sources are not initialized.
        RecordsWritePolicyError: If path policy validation fails.
    """
    audit_record_obj: Any = audit_record
    if not isinstance(audit_record_obj, dict):
        # audit_record 类型不符合预期，必须 fail-fast。
        raise TypeError("audit_record must be dict")
    audit_record_mapping = cast(Dict[str, Any], audit_record_obj)
    
    ctx = _require_fact_sources_initialized()
    if audit_records_dir is None:
        audit_records_dir = ctx.artifacts_dir / "path_audits"
    
    audit_records_dir.mkdir(parents=True, exist_ok=True)
    
    # 构造唯一的审计文件名：使用目标路径的哈希+时间戳避免重复。
    import hashlib
    target_path_str = str(audit_record_mapping.get("target_path", "unknown"))
    target_hash = hashlib.sha256(target_path_str.encode()).hexdigest()[:8]
    timestamp_raw = audit_record_mapping.get("timestamp", "")
    timestamp_str = str(timestamp_raw).replace(":", "-").replace(".", "-")
    audit_filename = f"path_audit_{target_hash}_{timestamp_str}.json"
    audit_path = audit_records_dir / audit_filename
    
    # 显式路径策略校验：禁止逃逸。
    path_policy.validate_output_target(audit_path, "artifact", ctx.run_root)
    
    # 写入审计记录，不触发门禁，使用 write_artifact_json_unbound。
    write_artifact_json_unbound(
        ctx.run_root,
        ctx.artifacts_dir,
        str(audit_path),
        audit_record_mapping
    )
    
    return audit_path


def _atomic_replace_write_bytes(dst: Path, data: bytes) -> None:
    """
    功能：原子写盘：同目录临时文件 + fsync + replace。
    
    Atomic write: create temp file in same directory, fsync, then replace target.
    Guarantees: gate validation must have completed before this function is called.
    On error, cleans up temp file.
    
    Args:
        dst: Destination file path.
        data: Bytes to write.
    
    Raises:
        (re-raises any exception from fsync or replace, after cleanup)
    """
    # 确保父目录存在
    _ensure_parent_dir(dst)
    
    # 在同目录创建临时文件
    # 使用 NamedTemporaryFile，delete=False，写完后手动管理
    fd = None
    tmp_path = None
    try:
        # 在目标目录中创建临时文件，确保同文件系统，便于原子 rename。
        fd, tmp_path_str = tempfile.mkstemp(dir=str(dst.parent), prefix=".tmp-", suffix=".writing")
        tmp_path = Path(tmp_path_str)
        
        # 写入 bytes
        os.write(fd, data)
        
        # flush + fsync
        os.fsync(fd)
        
        # 关闭 fd 再进行 replace
        os.close(fd)
        fd = None
        
        # 原子替换目标文件
        try:
            tmp_path.replace(dst)
        except PermissionError as replace_exc:
            if not _is_windows_winerror_5(replace_exc):
                raise

            # Windows 权限环境下 WinError 5 的受限回退：
            # 不再依赖 rename/replace，直接覆盖写入目标文件。
            fallback_fd = None
            try:
                if dst.exists():
                    try:
                        os.chmod(dst, 0o666)
                    except OSError:
                        pass

                fallback_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_BINARY"):
                    fallback_flags |= os.O_BINARY

                fallback_fd = os.open(str(dst), fallback_flags, 0o666)
                os.write(fallback_fd, data)
                os.fsync(fallback_fd)
            finally:
                if fallback_fd is not None:
                    os.close(fallback_fd)
        
    except Exception:
        # 非目标异常必须透传，禁止吞异常。
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        raise
    finally:
        # 统一清理 .writing 临时文件，覆盖主路径成功/WinError5 回退/异常分支。
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except PermissionError:
                try:
                    os.chmod(tmp_path, 0o666)
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    # 临时文件清理失败不应覆盖原始异常语义。
                    pass
            except Exception:
                # 临时文件清理失败不应覆盖原始异常语义。
                pass


def _require_fact_sources_initialized() -> FactSourcesContext:
    """
    功能：校验事实源是否已初始化。

    Require fact sources to be initialized before gate execution.

    Args:
        None.

    Returns:
        FactSourcesContext instance.

    Raises:
        FactSourcesNotInitializedError: If fact sources are missing.
    """
    ctx = _FACT_SOURCES_CONTEXT.get()
    if ctx is None:
        raise FactSourcesNotInitializedError(
            "Fact sources not initialized for records write: "
            "use records_io.bound_fact_sources() context manager."
        )
    return ctx


def _require_records_dir_initialized(path: Path) -> Path:
    """
    功能：要求 records_dir 已初始化。

    Require records_dir for path classification.

    Args:
        path: Target path.

    Returns:
        Resolved records_dir.

    Raises:
        RecordsWritePolicyError: If records_dir is not initialized.
    """
    ctx = _require_fact_sources_initialized()
    return ctx.records_dir


def _classify_output_kind(path: Path) -> str:
    """
    功能：判定写盘目标语义类型。

    Classify output target kind based on bound run layout.

    Args:
        path: Target path to classify.

    Returns:
        Output kind string: record, artifact, or log.

    Raises:
        RecordsWritePolicyError: If target escapes the bound layout.
    """
    path_obj: Any = path
    if not isinstance(path_obj, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")

    ctx = _require_fact_sources_initialized()
    normalized_path = path_obj
    target = normalized_path.resolve()
    if _is_under_dir(target, ctx.records_dir):
        return "record"
    if _is_under_dir(target, ctx.artifacts_dir):
        return "artifact"
    if _is_under_dir(target, ctx.logs_dir):
        return "log"
    raise RecordsWritePolicyError(
        "output path must be under run_root layout: "
        f"path={normalized_path}, run_root={ctx.run_root}"
    )


def _is_under_dir(target: Path, base_dir: Path) -> bool:
    """
    功能：判断路径是否在指定目录内。

    Check whether target is under base_dir using resolved paths.

    Args:
        target: Target path.
        base_dir: Base directory.

    Returns:
        True if target is under base_dir; otherwise False.
    """
    try:
        target.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return False
    return True


def _is_records_target(path: Path) -> bool:
    """
    功能：判断是否为 records 写盘目标。

    Determine whether a path is a records output target.

    Args:
        path: Target path.

    Returns:
        True if path is under records_dir and is .json/.jsonl.
    """
    if path.suffix not in {".json", ".jsonl"}:
        return False
    records_dir = _require_records_dir_initialized(path)
    return _is_under_dir(path, records_dir)


def _validate_record_for_records_path(record: Any, path: Path) -> None:
    """
    功能：执行 records 路径级门禁校验。

    Enforce records write policy: dict-only and gate validation.

    Args:
        record: Record object to validate.
        path: Target output path.

    Raises:
        RecordsWritePolicyError: If record is not dict or gate fails.
        FactSourcesNotInitializedError: If fact sources are missing.
    """
    if not isinstance(record, dict):
        # records 写盘必须为 dict，必须 fail-fast。
        raise RecordsWritePolicyError(
            "records write requires dict: "
            f"path={path}, actual_type={type(record).__name__}"
        )

    validated_record = cast(Dict[str, Any], record)

    ctx = _require_fact_sources_initialized()
    interpretation = get_contract_interpretation(ctx.contracts)
    schema.validate_record(validated_record, interpretation=interpretation)
    recommendations = assert_prewrite(
        validated_record,
        ctx.contracts,
        ctx.whitelist,
        ctx.semantics
    )
    _update_recommended_enforce_report(recommendations)


def _is_json_scalar(value: Any) -> bool:
    """
    功能：判断是否为 JSON 标量。

    Check whether a value is a JSON scalar.

    Args:
        value: Value to check.

    Returns:
        True if value is JSON scalar; otherwise False.
    """
    return isinstance(value, (str, int, float, bool)) or value is None


def _collect_json_keys(value: Any) -> set[str]:
    """
    功能：收集 JSON 结构中的所有字段名。

    Collect all key names from a JSON-like structure.

    Args:
        value: JSON-like value to traverse.

    Returns:
        Set of key names found in nested mappings.

    Raises:
        TypeError: If value contains non-JSON-like types or non-str keys.
    """
    if _is_json_scalar(value):
        return set()
    if not isinstance(value, (dict, list)):
        # value 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact payload must be JSON-like")

    keys: set[str] = set()
    stack: List[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            current_mapping = cast(Dict[Any, Any], current)
            for raw_key, raw_item in current_mapping.items():
                key_obj: Any = raw_key
                if not isinstance(key_obj, str):
                    # JSON key 类型不合法，必须 fail-fast。
                    raise TypeError("artifact payload keys must be str")
                keys.add(key_obj)
                if isinstance(raw_item, (dict, list)):
                    stack.append(raw_item)
        elif isinstance(current, list):
            current_items = cast(List[Any], current)
            for item in current_items:
                if isinstance(item, (dict, list)):
                    stack.append(item)
        else:
            # 非 JSON-like 类型，必须 fail-fast。
            raise TypeError("artifact payload must be JSON-like")
    return keys


def _collect_top_level_keys(obj: Dict[str, Any]) -> set[str]:
    """
    功能：收集顶层字段名集合。

    Collect top-level key names from a dict payload.

    Args:
        obj: Artifact payload mapping.

    Returns:
        Set of top-level key names.

    Raises:
        TypeError: If obj is not dict.
    """
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact obj must be dict")
    return set(cast(Dict[str, Any], obj_value).keys())


def _enforce_artifact_semantic_bypass_guard(
    obj: Dict[str, Any],
    dst_path: Path,
    artifacts_dir: Path,
    interpretation: ContractInterpretation | None
) -> None:
    """
    功能：阻断 artifacts 语义旁路。

    Enforce artifact semantic bypass guard using required record anchors.

    Args:
        obj: Artifact payload mapping.
        dst_path: Artifact output path.
        interpretation: Contract interpretation for required anchors.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        RecordsWritePolicyError: If artifact contains forbidden anchors.
    """
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact obj must be dict")
    dst_path_obj: Any = dst_path
    if not isinstance(dst_path_obj, Path):
        # dst_path 类型不符合预期，必须 fail-fast。
        raise TypeError("dst_path must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    interpretation_obj: Any = interpretation
    if interpretation_obj is not None and not isinstance(interpretation_obj, ContractInterpretation):
        # interpretation 类型不符合预期，必须 fail-fast。
        raise TypeError("interpretation must be ContractInterpretation or None")

    artifact_obj = cast(Dict[str, Any], obj_value)
    normalized_dst_path = dst_path_obj
    normalized_artifacts_dir = artifacts_dir_obj
    normalized_interpretation = interpretation_obj

    base_dir = normalized_artifacts_dir.resolve()
    target = normalized_dst_path.resolve()
    try:
        relative_path = target.relative_to(base_dir).as_posix()
    except ValueError as exc:
        # artifacts_dir 约束检查失败，必须 fail-fast。
        raise RecordsWritePolicyError(
            "artifact output must be under artifacts_dir: "
            f"path={normalized_dst_path}, artifacts_dir={normalized_artifacts_dir}"
        ) from exc

    present_fields = _collect_json_keys(artifact_obj)
    top_level_fields = _collect_top_level_keys(artifact_obj) - {"_artifact_audit"}

    # path_audits/ 路径模式允许包含策略版本等审计字段。
    if relative_path.startswith("path_audits/") and relative_path.endswith(".json"):
        # path_audit 可以包含 schema_version、policy_path、policy_path_semantics_version、runtime_whitelist_version。
        # 但禁止包含 bound digest 字段。
        forbidden_overlap = present_fields.intersection(_ARTIFACT_FORBIDDEN_ANCHOR_FIELDS)
        if forbidden_overlap:
            raise RecordsWritePolicyError(
                "path_audit payload contains forbidden bound digest fields: "
                f"path={normalized_dst_path}, fields={sorted(forbidden_overlap)}"
            )
        # path_audit 作为路径策略审计证据，允许包含策略版本字段，不再进一步检查锚点重叠。
        return

    if relative_path in _ARTIFACT_ANCHOR_ALLOWLIST:
        if relative_path == "cfg_audit/cfg_audit.json":
            top_level_fields = _collect_top_level_keys(artifact_obj)
            extra_top_level = top_level_fields - _ARTIFACT_CFG_AUDIT_ALLOWED_FIELDS
            if extra_top_level:
                raise RecordsWritePolicyError(
                    "cfg_audit payload contains non-allowlisted top-level fields: "
                    f"path={normalized_dst_path}, fields={sorted(extra_top_level)}"
                )
            # cfg_audit 不是 record，无需检查 required_record_fields
            # _artifact_audit 是系统注入的审计子字段，不作为顶级锚点考虑
        return

    forbidden_overlap = present_fields.intersection(_ARTIFACT_FORBIDDEN_ANCHOR_FIELDS)
    if forbidden_overlap:
        # artifacts 携带绑定摘要会形成语义旁路，必须 fail-fast。
        raise RecordsWritePolicyError(
            "artifact payload contains forbidden anchor fields: "
            f"path={normalized_dst_path}, fields={sorted(forbidden_overlap)}"
        )

    if normalized_interpretation is not None:
        required_fields = set(normalized_interpretation.required_record_fields)
        if required_fields:
            overlap = top_level_fields.intersection(required_fields)
            if overlap:
                # artifacts 具备 records 锚点字段，必须 fail-fast。
                raise RecordsWritePolicyError(
                    "artifact payload contains record anchor fields: "
                    f"path={normalized_dst_path}, fields={sorted(overlap)}"
                )

    ctx = _require_fact_sources_initialized()
    _enforce_governed_artifact_contract(
        artifact_obj,
        normalized_dst_path,
        relative_path,
        ctx.contracts,
    )


def write_json(
    path: str,
    obj: Any,
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    功能：写 JSON 文件：覆盖模式，原子写盘。
    
    Write object to JSON file (overwrite mode) with atomic semantics.
    Triggers gate validation if obj is a dict record.
    Uses atomic write: temp file in same directory, then replace.
    
    Args:
        path: Output file path.
        obj: Object to serialize (if dict, triggers validation).
        indent: JSON indent level.
        ensure_ascii: Whether to ensure ASCII encoding.
    """
    dst_path = Path(path)
    kind = _classify_output_kind(dst_path)
    if kind != "record":
        # records 写盘入口不得写入非 records 目录。
        raise RecordsWritePolicyError(
            "write_json can only target records_dir: "
            f"path={dst_path}, kind={kind}"
        )

    path_policy.validate_output_target(dst_path, "record", _require_fact_sources_initialized().run_root)
    if _is_records_target(dst_path):
        # records 写盘必须触发门禁校验。
        _validate_record_for_records_path(obj, dst_path)
    
    # 序列化为字符串
    json_str = _json_dumps_stable(obj, indent, ensure_ascii, compact=False)
    data = json_str.encode("utf-8")
    
    # 原子写盘
    _atomic_replace_write_bytes(dst_path, data)


def write_artifact_json(
    path: str,
    obj: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    功能：写审计产物 JSON。
    
    Write audit artifact JSON without gate validation.
    Uses the same stable serialization and atomic write as record output.
    
    Args:
        path: Output file path.
        obj: Artifact object to serialize (must be dict).
        indent: JSON indent level.
        ensure_ascii: Whether to ensure ASCII encoding.
    """
    path_obj: Any = path
    if not isinstance(path_obj, str) or not path_obj:
        # path 输入不合法，必须 fail-fast。
        raise ValueError("path must be non-empty str")
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact obj must be dict")
    indent_obj: Any = indent
    if not isinstance(indent_obj, int) or indent_obj < 0:
        # indent 输入不合法，必须 fail-fast。
        raise ValueError("indent must be non-negative int")
    ensure_ascii_obj: Any = ensure_ascii
    if not isinstance(ensure_ascii_obj, bool):
        # ensure_ascii 输入不合法，必须 fail-fast。
        raise ValueError("ensure_ascii must be bool")

    normalized_path_str = path_obj
    artifact_obj = cast(Dict[str, Any], obj_value)
    normalized_indent = indent_obj
    normalized_ensure_ascii = ensure_ascii_obj

    ctx = _require_fact_sources_initialized()
    dst_path = Path(normalized_path_str)
    kind = _classify_output_kind(dst_path)
    if kind != "artifact":
        # artifact 写盘必须隔离到 artifacts 目录。
        raise RecordsWritePolicyError(
            "artifact output must be under artifacts_dir: "
            f"path={dst_path}, kind={kind}"
        )
    path_policy.validate_output_target(dst_path, "artifact", ctx.run_root)
    interpretation = get_contract_interpretation(ctx.contracts)
    _ensure_artifact_audit_marker(artifact_obj)
    _enforce_artifact_semantic_bypass_guard(artifact_obj, dst_path, ctx.artifacts_dir, interpretation)
    json_str = _json_dumps_stable(artifact_obj, normalized_indent, normalized_ensure_ascii, compact=False)
    data = json_str.encode("utf-8")

    _atomic_replace_write_bytes(dst_path, data)


def write_artifact_json_unbound(
    run_root: Path,
    artifacts_dir: Path,
    path: str,
    obj: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    功能：无 fact sources 上下文的 artifact 兜底写盘。
    
    Write artifact JSON without requiring fact sources context.
    Used as fallback when fact sources initialization fails before run_closure can be written.
    
    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts output directory.
        path: Output file path (string, typically artifacts_dir/run_closure.json).
        obj: Artifact object to serialize (must be dict).
        indent: JSON indent level.
        ensure_ascii: Whether to ensure ASCII encoding.
    
    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs are structurally invalid.
        RecordsWritePolicyError: If path escapes artifacts_dir or other policy violations.
    """
    # 输入类型校验
    run_root_obj: Any = run_root
    if not isinstance(run_root_obj, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    path_obj: Any = path
    if not isinstance(path_obj, str) or not path_obj:
        # path 输入不合法，必须 fail-fast。
        raise ValueError("path must be non-empty str")
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact obj must be dict")
    indent_obj: Any = indent
    if not isinstance(indent_obj, int) or indent_obj < 0:
        # indent 输入不合法，必须 fail-fast。
        raise ValueError("indent must be non-negative int")
    ensure_ascii_obj: Any = ensure_ascii
    if not isinstance(ensure_ascii_obj, bool):
        # ensure_ascii 输入不合法，必须 fail-fast。
        raise ValueError("ensure_ascii must be bool")

    normalized_run_root = run_root_obj
    normalized_artifacts_dir = artifacts_dir_obj
    normalized_path_str = path_obj
    artifact_obj = cast(Dict[str, Any], obj_value)
    normalized_indent = indent_obj
    normalized_ensure_ascii = ensure_ascii_obj
    
    # 规范化路径。
    run_root_resolved = normalized_run_root.resolve()
    artifacts_dir_resolved = normalized_artifacts_dir.resolve()
    records_dir_resolved = (run_root_resolved / "records").resolve()
    dst_path = Path(normalized_path_str)
    
    # 若 path 非绝对路径，则相对于 run_root 解释。
    if not dst_path.is_absolute():
        dst_path = run_root_resolved / dst_path
    dst_path_resolved = dst_path.resolve()
    
    # 硬约束：禁止向 records 目录写入（防止语义旁路）。
    try:
        dst_path_resolved.relative_to(records_dir_resolved)
        # 目标路径在 records 目录内，必须 fail-fast。
        raise RecordsWritePolicyError(
            "unbound artifact output must not write to records_dir (semantic bypass): "
            f"path={dst_path}, records_dir={records_dir_resolved}, run_root={normalized_run_root}"
        )
    except ValueError:
        # 正常情况：目标路径不在 records 目录内，继续执行。
        pass
    
    # 路径约束检查：确保写入目标在 artifacts_dir 子树内。
    try:
        dst_path_resolved.relative_to(artifacts_dir_resolved)
    except ValueError:
        # artifacts_dir 约束检查失败，目录逃逸，必须 fail-fast。
        raise RecordsWritePolicyError(
            "unbound artifact output must be under artifacts_dir: "
            f"path={dst_path}, artifacts_dir={normalized_artifacts_dir}, run_root={normalized_run_root}"
        )
    
    # 尽量使用 path_policy 验证。
    try:
        path_policy.validate_output_target(dst_path_resolved, "artifact", run_root_resolved)
    except Exception as exc:
        # path_policy 验证失败，必须 fail-fast。
        raise RecordsWritePolicyError(
            f"unbound artifact output failed path_policy validation: path={dst_path}, error={exc}"
        )
    
    # 确保父目录存在。
    _ensure_parent_dir(dst_path_resolved)
    
    # 添加审计标记。
    _ensure_artifact_audit_marker(artifact_obj)

    # 序列化为字符串。
    _enforce_artifact_semantic_bypass_guard(artifact_obj, dst_path_resolved, artifacts_dir_resolved, None)
    json_str = _json_dumps_stable(artifact_obj, normalized_indent, normalized_ensure_ascii, compact=False)
    data = json_str.encode("utf-8")
    
    # 原子写盘。
    _atomic_replace_write_bytes(dst_path_resolved, data)


def write_artifact_canon_json_unbound(
    run_root: Path,
    artifacts_dir: Path,
    path: str,
    obj: Dict[str, Any]
) -> None:
    """
    功能：无 fact sources 上下文的 artifact canonical JSON 写盘。

    Write artifact canonical JSON without requiring fact sources context.
    Canonical JSON uses fixed separators and sorted keys.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts output directory.
        path: Output file path (string, typically under artifacts_dir).
        obj: Artifact object to serialize (must be dict).

    Returns:
        None.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs are structurally invalid.
        RecordsWritePolicyError: If path escapes artifacts_dir or other policy violations.
    """
    run_root_obj: Any = run_root
    if not isinstance(run_root_obj, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    path_obj: Any = path
    if not isinstance(path_obj, str) or not path_obj:
        # path 输入不合法，必须 fail-fast。
        raise ValueError("path must be non-empty str")
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact obj must be dict")

    normalized_run_root = run_root_obj
    normalized_artifacts_dir = artifacts_dir_obj
    normalized_path_str = path_obj
    artifact_obj = cast(Dict[str, Any], obj_value)

    run_root_resolved = normalized_run_root.resolve()
    artifacts_dir_resolved = normalized_artifacts_dir.resolve()
    records_dir_resolved = (run_root_resolved / "records").resolve()
    dst_path = Path(normalized_path_str)

    if not dst_path.is_absolute():
        dst_path = run_root_resolved / dst_path
    dst_path_resolved = dst_path.resolve()

    try:
        dst_path_resolved.relative_to(records_dir_resolved)
        # 目标路径在 records 目录内，必须 fail-fast。
        raise RecordsWritePolicyError(
            "unbound artifact canonical output must not write to records_dir (semantic bypass): "
            f"path={dst_path}, records_dir={records_dir_resolved}, run_root={normalized_run_root}"
        )
    except ValueError:
        pass

    try:
        dst_path_resolved.relative_to(artifacts_dir_resolved)
    except ValueError:
        # artifacts_dir 约束检查失败，目录逃逸，必须 fail-fast。
        raise RecordsWritePolicyError(
            "unbound artifact canonical output must be under artifacts_dir: "
            f"path={dst_path}, artifacts_dir={normalized_artifacts_dir}, run_root={normalized_run_root}"
        )

    try:
        path_policy.validate_output_target(dst_path_resolved, "artifact", run_root_resolved)
    except Exception as exc:
        # path_policy 验证失败，必须 fail-fast。
        raise RecordsWritePolicyError(
            f"unbound artifact canonical output failed path_policy validation: path={dst_path}, error={exc}"
        ) from exc

    _ensure_parent_dir(dst_path_resolved)
    _ensure_artifact_audit_marker(artifact_obj)
    _enforce_artifact_semantic_bypass_guard(artifact_obj, dst_path_resolved, artifacts_dir_resolved, None)

    from main.core import digests
    data = digests.canonical_json_dumps(artifact_obj)
    _atomic_replace_write_bytes(dst_path_resolved, data)


def write_artifact_text_unbound(
    run_root: Path,
    artifacts_dir: Path,
    path: str,
    content: str
) -> None:
    """
    功能：无 fact sources 上下文的 artifact 文本写盘。

    Write artifact text without requiring fact sources context.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts output directory.
        path: Output file path (string, typically under artifacts_dir).
        content: Text content to write.

    Returns:
        None.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs are structurally invalid.
        RecordsWritePolicyError: If path escapes artifacts_dir or other policy violations.
    """
    run_root_obj: Any = run_root
    if not isinstance(run_root_obj, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    path_obj: Any = path
    if not isinstance(path_obj, str) or not path_obj:
        # path 输入不合法，必须 fail-fast。
        raise ValueError("path must be non-empty str")
    content_obj: Any = content
    if not isinstance(content_obj, str):
        # content 类型不合法，必须 fail-fast。
        raise TypeError("content must be str")

    # 显式路径策略校验：禁止逃逸。
    path_policy.validate_output_target(Path(path_obj), "artifact", run_root_obj)

    data = content_obj.encode("utf-8")
    write_artifact_bytes_unbound(run_root_obj, artifacts_dir_obj, path_obj, data)


def write_artifact_bytes_unbound(
    run_root: Path,
    artifacts_dir: Path,
    path: str,
    data: bytes
) -> None:
    """
    功能：无 fact sources 上下文的 artifact bytes 写盘。

    Write artifact bytes without requiring fact sources context.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts output directory.
        path: Output file path (string, typically under artifacts_dir).
        data: Bytes content to write.

    Returns:
        None.

    Raises:
        TypeError: If inputs are of invalid types.
        ValueError: If inputs are structurally invalid.
        RecordsWritePolicyError: If path escapes artifacts_dir or other policy violations.
    """
    run_root_obj: Any = run_root
    if not isinstance(run_root_obj, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    path_obj: Any = path
    if not isinstance(path_obj, str) or not path_obj:
        # path 输入不合法，必须 fail-fast。
        raise ValueError("path must be non-empty str")
    if not isinstance(data, (bytes, bytearray)):
        # data 类型不合法，必须 fail-fast。
        raise TypeError("data must be bytes")

    # 显式路径策略校验：禁止逃逸。
    normalized_run_root = run_root_obj
    normalized_artifacts_dir = artifacts_dir_obj
    normalized_path_str = path_obj
    path_policy.validate_output_target(Path(normalized_path_str), "artifact", normalized_run_root)

    run_root_resolved = normalized_run_root.resolve()
    artifacts_dir_resolved = normalized_artifacts_dir.resolve()
    records_dir_resolved = (run_root_resolved / "records").resolve()
    dst_path = Path(normalized_path_str)

    if not dst_path.is_absolute():
        dst_path = run_root_resolved / dst_path
    dst_path_resolved = dst_path.resolve()

    try:
        dst_path_resolved.relative_to(records_dir_resolved)
        # 目标路径在 records 目录内，必须 fail-fast。
        raise RecordsWritePolicyError(
            "unbound artifact bytes output must not write to records_dir (semantic bypass): "
            f"path={dst_path}, records_dir={records_dir_resolved}, run_root={normalized_run_root}"
        )
    except ValueError:
        pass

    try:
        dst_path_resolved.relative_to(artifacts_dir_resolved)
    except ValueError:
        # artifacts_dir 约束检查失败，目录逃逸，必须 fail-fast。
        raise RecordsWritePolicyError(
            "unbound artifact bytes output must be under artifacts_dir: "
            f"path={dst_path}, artifacts_dir={normalized_artifacts_dir}, run_root={normalized_run_root}"
        )

    try:
        path_policy.validate_output_target(dst_path_resolved, "artifact", run_root_resolved)
    except Exception as exc:
        # path_policy 验证失败，必须 fail-fast。
        raise RecordsWritePolicyError(
            f"unbound artifact bytes output failed path_policy validation: path={dst_path}, error={exc}"
        ) from exc

    _ensure_parent_dir(dst_path_resolved)
    _atomic_replace_write_bytes(dst_path_resolved, bytes(data))


def append_jsonl(
    path: str,
    record: Dict[str, Any]
) -> None:
    """
    功能：追加 JSONL 记录。
    
    Append record to JSONL file with line-level fsync.
    Triggers gate validation before write.
    
    Args:
        path: Output JSONL file path.
        record: Record dict to append.
    """
    dst_path = Path(path)
    kind = _classify_output_kind(dst_path)
    if kind != "record":
        # records 写盘入口不得写入非 records 目录。
        raise RecordsWritePolicyError(
            "append_jsonl can only target records_dir: "
            f"path={dst_path}, kind={kind}"
        )
    path_policy.validate_output_target(dst_path, "record", _require_fact_sources_initialized().run_root)
    if _is_records_target(dst_path):
        # records 写盘必须触发门禁校验。
        _validate_record_for_records_path(record, dst_path)
    else:
        record_obj: Any = record
        if not isinstance(record_obj, dict):
        # record 类型不符合预期，必须 fail-fast。
            raise RecordsWritePolicyError(
                "jsonl record must be dict: "
                f"path={dst_path}, actual_type={type(record_obj).__name__}"
            )
    
    # 确保父目录存在。
    _ensure_parent_dir(dst_path)
    
    # 打开文件追加写，并在一行写完后 fsync。
    # 使用 os.open 获取 fd，便于手动 fsync。
    fd = None
    try:
        # 打开文件追加，若不存在则创建。
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(dst_path), flags, 0o644)
        
        # 序列化为紧凑单行 JSON。
        json_line = _json_dumps_stable(record, None, False, compact=True)
        line_bytes = (json_line + "\n").encode("utf-8")
        
        # 写入一行。
        os.write(fd, line_bytes)
        
        # fsync 确保写入磁盘。
        os.fsync(fd)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def write_text(
    path: str,
    content: str,
    record_ctx: Optional[Dict[str, Any]] = None
) -> None:
    """
    功能：写文本文件。
    
    Write text to file (overwrite mode) with atomic semantics.
    Triggers gate validation if record_ctx is provided.
    Uses atomic write: temp file in same directory, then replace.
    
    Args:
        path: Output file path.
        content: Text content to write.
        record_ctx: Optional record context for gate validation.
                    If provided and fact sources are loaded, triggers validation.
    """
    ctx = _require_fact_sources_initialized()
    dst_path = Path(path)
    kind = _classify_output_kind(dst_path)
    if kind == "record":
        # write_text 禁止写入 records 目录，必须 fail-fast。
        raise RecordsWritePolicyError(
            "write_text cannot target records_dir: "
            f"path={dst_path}"
        )
    if kind not in {"artifact", "log"}:
        # write_text 仅允许 artifacts 或 logs。
        raise RecordsWritePolicyError(
            "write_text must target artifacts_dir or logs_dir: "
            f"path={dst_path}, kind={kind}"
        )
    path_policy.validate_output_target(dst_path, kind, ctx.run_root)

    # 先门禁校验。
    if record_ctx is not None:
        interpretation = get_contract_interpretation(ctx.contracts)
        schema.validate_record(record_ctx, interpretation=interpretation)
        recommendations = assert_prewrite(
            record_ctx,
            ctx.contracts,
            ctx.whitelist,
            ctx.semantics
        )
        _update_recommended_enforce_report(recommendations)

    # 编码为字符串内容。
    data = content.encode("utf-8")

    # 原子写盘。
    _atomic_replace_write_bytes(dst_path, data)


def _ensure_artifact_audit_marker(obj: Dict[str, Any]) -> None:
    """
    功能：为 artifacts 添加最小审计标识。

    Add minimal audit marker for artifact payloads.

    Args:
        obj: Artifact payload mapping.

    Returns:
        None.

    Raises:
        TypeError: If obj or marker types are invalid.
        ValueError: If marker fields are invalid.
    """
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("artifact obj must be dict")

    artifact_obj = cast(Dict[str, Any], obj_value)

    marker_key = "_artifact_audit"
    if marker_key not in artifact_obj:
        artifact_obj[marker_key] = {
            "schema_version": "v1.0",
            "writer": "records_io"
        }
        return

    marker = artifact_obj.get(marker_key)
    if not isinstance(marker, dict):
        # marker 类型不符合预期，必须 fail-fast。
        raise TypeError("_artifact_audit must be dict")
    marker_mapping = cast(Dict[str, Any], marker)
    schema_version = marker_mapping.get("schema_version")
    writer = marker_mapping.get("writer")
    if not isinstance(schema_version, str) or not schema_version:
        # schema_version 非法，必须 fail-fast。
        raise ValueError("_artifact_audit.schema_version must be non-empty str")
    if not isinstance(writer, str) or not writer:
        # writer 非法，必须 fail-fast。
        raise ValueError("_artifact_audit.writer must be non-empty str")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    功能：读取 JSONL 文件。
    
    Read all records from JSONL file.
    
    Args:
        path: Input JSONL file path.
    
    Returns:
        List of parsed records.
    """
    path_obj = Path(path)
    records: List[Dict[str, Any]] = []
    
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    return records


def read_json(path: str) -> Any:
    """
    功能：读取 JSON 文件。
    
    Read object from JSON file.
    
    Args:
        path: Input JSON file path.
    
    Returns:
        Parsed JSON object.
    """
    path_obj = Path(path)
    
    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_file_controlled(src_path: Path, dst_path: Path, kind: str = "artifact") -> None:
    """
    功能：受控文件复制（冻结门禁验证）。
    
    Copy file from src to dst with gate enforcement and output path validation.
    Must be called within bound_fact_sources() context. All copies routed through
    validate_output_target() to ensure destination respects output layout policy.
    Copied file receives no hash anchoring (caller responsible if needed).
    
    Args:
        src_path: Source file path (absolute or relative).
        dst_path: Destination file path (will be validated against layout).
        kind: Output kind for validation ("artifact", "record", "log", "env_lock").
    
    Returns:
        None.
    
    Raises:
        TypeError: If inputs are invalid types.
        ValueError: If kind is unsupported.
        FileNotFoundError: If source file does not exist.
        RecordsWritePolicyError: If destination violates layout constraints.
        GateEnforcementError: If gate enforcement fails.
    """
    src_path_obj: Any = src_path
    if not isinstance(src_path_obj, Path):
        # src_path 类型不符合预期，必须 fail-fast。
        raise TypeError("src_path must be Path")
    dst_path_obj: Any = dst_path
    if not isinstance(dst_path_obj, Path):
        # dst_path 类型不符合预期，必须 fail-fast。
        raise TypeError("dst_path must be Path")
    kind_obj: Any = kind
    if not isinstance(kind_obj, str) or kind_obj not in ("artifact", "record", "log", "env_lock"):
        # kind 输入不合法，必须 fail-fast。
        raise ValueError(f"kind must be 'artifact', 'record', 'log', or 'env_lock', got {kind_obj}")

    normalized_src_path = src_path_obj
    normalized_dst_path = dst_path_obj
    normalized_kind = kind_obj
    
    ctx = _require_fact_sources_initialized()
    
    if not normalized_src_path.exists() or not normalized_src_path.is_file():
        # 源文件缺失，必须 fail-fast。
        raise FileNotFoundError(f"Source file not found: {normalized_src_path}")
    
    # 校验输出目标路径。
    if normalized_kind == "env_lock":
        expected_lock_path = ctx.run_root / "requirements.txt"
        if normalized_dst_path.resolve() != expected_lock_path.resolve():
            raise RecordsWritePolicyError(
                "env_lock output must be run_root/requirements.txt: "
                f"path={normalized_dst_path}, expected={expected_lock_path}"
            )
    else:
        path_policy.validate_output_target(normalized_dst_path, normalized_kind, ctx.run_root)
    
    # 原子性复制（使用 shutil.copy2 保留时间戳）。
    try:
        import shutil
        normalized_dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(normalized_src_path, normalized_dst_path)
    except Exception as exc:
        # 复制操作失败，必须 fail-fast。
        raise OSError(
            f"Failed to copy {normalized_src_path} to {normalized_dst_path}: {exc}"
        ) from exc


def copy_file_controlled_unbound(
    run_root: Path,
    artifacts_dir: Path,
    src_path: Path,
    dst_path: Path,
    kind: str = "artifact"
) -> None:
    """
    功能：无上下文受控文件复制（冻结门禁路径校验）。

    Copy file with explicit run_root/artifacts_dir context, without relying on
    global bound_fact_sources state. This interface is intended for fallback
    phases where fact sources may be unavailable.

    Args:
        run_root: Run root directory.
        artifacts_dir: Artifacts directory under run_root.
        src_path: Source file path.
        dst_path: Destination file path.
        kind: Output kind ("artifact", "record", "log", "env_lock").

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If kind is unsupported.
        FileNotFoundError: If source file does not exist.
        RecordsWritePolicyError: If destination violates policy constraints.
        OSError: If copy fails.
    """
    run_root_obj: Any = run_root
    if not isinstance(run_root_obj, Path):
        # run_root 类型不符合预期，必须 fail-fast。
        raise TypeError("run_root must be Path")
    artifacts_dir_obj: Any = artifacts_dir
    if not isinstance(artifacts_dir_obj, Path):
        # artifacts_dir 类型不符合预期，必须 fail-fast。
        raise TypeError("artifacts_dir must be Path")
    src_path_obj: Any = src_path
    if not isinstance(src_path_obj, Path):
        # src_path 类型不符合预期，必须 fail-fast。
        raise TypeError("src_path must be Path")
    dst_path_obj: Any = dst_path
    if not isinstance(dst_path_obj, Path):
        # dst_path 类型不符合预期，必须 fail-fast。
        raise TypeError("dst_path must be Path")
    kind_obj: Any = kind
    if not isinstance(kind_obj, str) or kind_obj not in ("artifact", "record", "log", "env_lock"):
        # kind 输入不合法，必须 fail-fast。
        raise ValueError(f"kind must be 'artifact', 'record', 'log', or 'env_lock', got {kind_obj}")

    normalized_run_root: Path = run_root_obj
    normalized_src_path: Path = src_path_obj
    normalized_dst_path: Path = dst_path_obj
    normalized_kind: str = kind_obj

    if not normalized_src_path.exists() or not normalized_src_path.is_file():
        # 源文件缺失，必须 fail-fast。
        raise FileNotFoundError(f"Source file not found: {normalized_src_path}")

    if normalized_kind == "env_lock":
        expected_lock_path = normalized_run_root / "requirements.txt"
        if normalized_dst_path.resolve() != expected_lock_path.resolve():
            raise RecordsWritePolicyError(
                "env_lock output must be run_root/requirements.txt: "
                f"path={normalized_dst_path}, expected={expected_lock_path}"
            )
    else:
        path_policy.validate_output_target(normalized_dst_path, normalized_kind, normalized_run_root)

    try:
        import shutil
        normalized_dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(normalized_src_path, normalized_dst_path)
    except Exception as exc:
        # 复制操作失败，必须 fail-fast。
        raise OSError(
            f"Failed to copy {normalized_src_path} to {normalized_dst_path}: {exc}"
        ) from exc
