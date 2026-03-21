"""
CLI 入口层公共辅助逻辑

功能说明：
- 提供 CLI 入口层共享的冻结面相关辅助函数，避免入口分叉导致的语义漂移。
"""

import os
from typing import Any, Dict, Optional, Tuple, cast
from main.core.contracts import FrozenContracts, get_contract_interpretation
from main.core.errors import MissingRequiredFieldError, RunFailureReason
from main.registries import runtime_resolver
from main.core import digests
from main.diffusion.sd3.callback_composer import InjectionContext
from main.watermarking.content_chain import channel_lf, channel_hf
from main.watermarking.provenance.key_derivation import resolve_attestation_event_binding_mode
from main.core import time_utils


_SEED_RULE_ID = "stable_seed_from_parts_v1"
_REQUIRED_SEED_PART_KEYS = {"key_id", "sample_idx", "purpose"}


def _resolve_mapping(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    child = parent.get(key)
    return cast(Dict[str, Any], child) if isinstance(child, dict) else {}


def _resolve_plan_basis_digest(plan_payload: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    """
    功能：解析注入期 basis_digest。

    Resolve the authoritative basis digest for injection-time runtime.

    Args:
        plan_payload: Planner payload mapping.
        cfg: Configuration mapping.

    Returns:
        Basis digest when available.
    """
    direct_candidate = plan_payload.get("basis_digest")
    if isinstance(direct_candidate, str) and direct_candidate:
        return direct_candidate
    lf_basis = plan_payload.get("lf_basis") if isinstance(plan_payload.get("lf_basis"), dict) else {}
    lf_basis_digest = lf_basis.get("basis_digest") or lf_basis.get("projection_matrix_digest")
    if isinstance(lf_basis_digest, str) and lf_basis_digest:
        return lf_basis_digest
    cfg_candidate = cfg.get("basis_digest") or cfg.get("lf_basis_digest")
    if isinstance(cfg_candidate, str) and cfg_candidate:
        return cfg_candidate
    if isinstance(lf_basis, dict) and lf_basis:
        return digests.canonical_sha256(lf_basis)
    return None


def _resolve_canonical_event_binding_mode(cfg: Dict[str, Any]) -> Optional[str]:
    """
    功能：解析注入链使用的规范化事件绑定模式。

    Resolve the canonical event-binding mode for the injection path.

    Args:
        cfg: Configuration mapping.

    Returns:
        Canonical event-binding mode when available; otherwise None.
    """
    direct_candidate = cfg.get("event_binding_mode")
    if isinstance(direct_candidate, str) and direct_candidate:
        return direct_candidate

    attestation_runtime = _resolve_mapping(cfg, "attestation_runtime")
    runtime_candidate = attestation_runtime.get("event_binding_mode")
    if isinstance(runtime_candidate, str) and runtime_candidate:
        return runtime_candidate

    watermark_cfg = _resolve_mapping(cfg, "watermark")
    watermark_candidate = watermark_cfg.get("event_binding_mode")
    if isinstance(watermark_candidate, str) and watermark_candidate:
        return watermark_candidate

    attestation_cfg = _resolve_mapping(cfg, "attestation")
    use_trajectory_mix = attestation_cfg.get("use_trajectory_mix")
    if isinstance(use_trajectory_mix, bool):
        return resolve_attestation_event_binding_mode(use_trajectory_mix)

    return None


def resolve_attestation_env_inputs(
    cfg: Any,
    *,
    require_prompt_seed: bool,
) -> Dict[str, Any]:
    """
    功能：从环境变量解析 attestation 主链所需的密钥输入。

    Resolve attestation secret inputs from environment variables for the main
    embed/detect CLI path. The resolved values are transient and must never be
    persisted to cfg_audit or records.

    Args:
        cfg: Configuration mapping.
        require_prompt_seed: Whether k_prompt and k_seed are required.

    Returns:
        Mapping with resolution status, env var names, missing fields, and any
        resolved secret values.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    cfg_dict = cast(Dict[str, Any], cfg)

    attestation_cfg = _resolve_mapping(cfg_dict, "attestation")
    if not bool(attestation_cfg.get("enabled", False)):
        return {
            "status": "absent",
            "attestation_absent_reason": "attestation_disabled",
        }

    k_master_env_var = str(attestation_cfg.get("k_master_env_var", "CEG_WM_K_MASTER"))
    k_prompt_env_var = str(attestation_cfg.get("k_prompt_env_var", "CEG_WM_K_PROMPT"))
    k_seed_env_var = str(attestation_cfg.get("k_seed_env_var", "CEG_WM_K_SEED"))

    result: Dict[str, Any] = {
        "status": "ok",
        "k_master_env_var": k_master_env_var,
        "k_prompt_env_var": k_prompt_env_var,
        "k_seed_env_var": k_seed_env_var,
    }

    k_master = os.environ.get(k_master_env_var, "")
    if k_master.strip():
        result["k_master"] = k_master.strip()

    if require_prompt_seed:
        k_prompt = os.environ.get(k_prompt_env_var, "")
        k_seed = os.environ.get(k_seed_env_var, "")
        if k_prompt.strip():
            result["k_prompt"] = k_prompt.strip()
        if k_seed.strip():
            result["k_seed"] = k_seed.strip()

    missing_secret_fields: list[str] = []
    if not isinstance(result.get("k_master"), str) or not result.get("k_master"):
        missing_secret_fields.append("k_master")
    if require_prompt_seed and (not isinstance(result.get("k_prompt"), str) or not result.get("k_prompt")):
        missing_secret_fields.append("k_prompt")
    if require_prompt_seed and (not isinstance(result.get("k_seed"), str) or not result.get("k_seed")):
        missing_secret_fields.append("k_seed")

    if missing_secret_fields:
        result["status"] = "absent"
        result["attestation_absent_reason"] = "attestation_secret_missing"
        result["missing_secret_fields"] = missing_secret_fields

    return result


def build_cli_config_migration_hint(exc: Any) -> Optional[str]:
    """
    功能：为配置校验错误生成 CLI 迁移提示。

    Build migration hint text for known config-validation errors.

    Args:
        exc: Raised exception from config loading or validation.

    Returns:
        Human-readable migration hint when matched; otherwise None.

    Raises:
        TypeError: If exc is invalid.
    """
    if not isinstance(exc, Exception):
        # exc 类型不合法，必须 fail-fast。
        raise TypeError("exc must be Exception")

    message = str(exc)
    if "paper_faithfulness requires watermark.lf.ecc='sparse_ldpc'" in message:
        return (
            "检测到 paper_faithfulness 模式下使用了 legacy int ecc。"
            "请将 watermark.lf.ecc 设置为 \"sparse_ldpc\"；"
            "如需继续使用 int 兼容分支，请先将 paper_faithfulness.enabled 设为 false。"
        )
    return None


def build_injection_context_from_plan(
    cfg: Any,
    plan_payload: Any,
    plan_digest: Any
) -> InjectionContext:
    """
    功能：基于 plan 构造 InjectionContext。
    
    Build InjectionContext from plan payload and cfg without mutating cfg.

    Args:
        cfg: Configuration mapping.
        plan_payload: Planner output mapping (plan payload).
        plan_digest: Plan digest string.

    Returns:
        InjectionContext instance.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If required fields are missing.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(plan_payload, dict):
        # plan_payload 类型不符合预期，必须 fail-fast。
        raise TypeError("plan_payload must be dict")
    if not isinstance(plan_digest, str) or not plan_digest:
        # plan_digest 类型不符合预期，必须 fail-fast。
        raise TypeError("plan_digest must be non-empty str")
    cfg_dict = cast(Dict[str, Any], cfg)
    plan_payload_dict = cast(Dict[str, Any], plan_payload)

    watermark_cfg = _resolve_mapping(cfg_dict, "watermark")
    lf_cfg = _resolve_mapping(watermark_cfg, "lf")
    hf_cfg = _resolve_mapping(watermark_cfg, "hf")

    enable_lf = bool(lf_cfg.get("enabled", False))
    enable_hf = bool(hf_cfg.get("enabled", False))

    lf_strength = lf_cfg.get("strength", cfg_dict.get("lf_strength", 1.5))
    hf_threshold_percentile = hf_cfg.get("threshold_percentile", cfg_dict.get("hf_threshold_percentile", 75.0))

    lf_params: Dict[str, Any] = {
        "impl_id": channel_lf.LF_CHANNEL_IMPL_ID,
        "impl_version": channel_lf.LF_CHANNEL_VERSION,
        "lf_strength": lf_strength,
        "lf_enabled": enable_lf
    }
    hf_params: Dict[str, Any] = {
        "impl_id": channel_hf.HF_CHANNEL_IMPL_ID,
        "impl_version": channel_hf.HF_CHANNEL_VERSION,
        "hf_threshold_percentile": hf_threshold_percentile,
        "hf_enabled": enable_hf
    }
    lf_params_digest = digests.canonical_sha256(lf_params) if enable_lf else ""
    hf_params_digest = digests.canonical_sha256(hf_params) if enable_hf else ""

    device = cfg_dict.get("device", "cpu")
    dtype = cfg_dict.get("dtype", "float32")
    model_cfg = _resolve_mapping(cfg_dict, "model")
    if "dtype" in model_cfg:
        dtype = model_cfg.get("dtype", dtype)

    attestation_runtime = _resolve_mapping(cfg_dict, "attestation_runtime")
    basis_digest = _resolve_plan_basis_digest(plan_payload_dict, cfg_dict)
    attestation_digest = cfg_dict.get("attestation_digest") or attestation_runtime.get("attestation_digest")
    attestation_event_digest = cfg_dict.get("attestation_event_digest") or attestation_runtime.get("event_binding_digest")
    lf_attestation_event_digest = cfg_dict.get("lf_attestation_event_digest") or attestation_event_digest
    lf_attestation_key = cfg_dict.get("lf_attestation_key") or cfg_dict.get("k_lf") or attestation_runtime.get("k_lf")
    event_binding_mode = _resolve_canonical_event_binding_mode(cfg_dict)

    return InjectionContext(
        plan_digest=plan_digest,
        plan_ref=plan_payload_dict,
        lf_params_digest=lf_params_digest,
        hf_params_digest=hf_params_digest,
        enable_lf=enable_lf,
        enable_hf=enable_hf,
        basis_digest=basis_digest if isinstance(basis_digest, str) and basis_digest else None,
        attestation_digest=attestation_digest if isinstance(attestation_digest, str) and attestation_digest else None,
        attestation_event_digest=(
            attestation_event_digest if isinstance(attestation_event_digest, str) and attestation_event_digest else None
        ),
        lf_attestation_event_digest=(
            lf_attestation_event_digest if isinstance(lf_attestation_event_digest, str) and lf_attestation_event_digest else None
        ),
        lf_attestation_key=lf_attestation_key if isinstance(lf_attestation_key, str) and lf_attestation_key else None,
        event_binding_mode=event_binding_mode if isinstance(event_binding_mode, str) and event_binding_mode else None,
        device=device if isinstance(device, str) and device else "cpu",
        dtype=dtype if isinstance(dtype, str) and dtype else "float32"
    )


def set_value_by_field_path(record: Dict[str, Any], field_path: str, value: str) -> None:
    """
    功能：按点路径写入 record 字段值。

    Set a nested record value by dotted field path.

    Args:
        record: Record dict to mutate.
        field_path: Dotted field path.
        value: Value to set.

    Returns:
        None.

    Raises:
        TypeError: If record is not dict.
        ValueError: If field_path or value is invalid.
    """
    if not field_path:
        # value 输入不合法，必须 fail-fast。
        raise ValueError("value must be non-empty str")

    current = record
    segments = field_path.split(".")
    for segment in segments[:-1]:
        if not segment:
            # field_path 段为空，必须 fail-fast。
            raise ValueError(f"Invalid field_path segment in {field_path}")
        if segment not in current or not isinstance(current[segment], dict):
            current[segment] = {}
        current = current[segment]
    last = segments[-1]
    if not last:
        # field_path 末段为空，必须 fail-fast。
        raise ValueError(f"Invalid field_path segment in {field_path}")
    current[last] = value


def bind_impl_identity_fields(
    record: Dict[str, Any],
    identity: runtime_resolver.ImplIdentity,
    impl_set: runtime_resolver.BuiltImplSet,
    contracts: FrozenContracts
) -> None:
    """
    功能：绑定 impl_identity 相关字段。

    Bind impl identity, version, and digest fields into record.

    Args:
        record: Record dict to mutate.
        identity: Impl identity mapping.
        impl_set: Built implementation set.
        contracts: Loaded FrozenContracts.

    Returns:
        None.

    Raises:
        MissingRequiredFieldError: If required impl fields are missing.
        TypeError: If inputs are invalid.
    """
    interpretation = get_contract_interpretation(contracts)
    impl_identity_spec = interpretation.impl_identity
    field_paths_by_domain = impl_identity_spec.field_paths_by_domain
    version_field_paths_by_domain = impl_identity_spec.version_field_paths_by_domain
    digest_field_paths_by_domain = impl_identity_spec.digest_field_paths_by_domain

    impl_id_by_domain = {
        "content_extractor": identity.content_extractor_id,
        "geometry_extractor": identity.geometry_extractor_id,
        "fusion_rule": identity.fusion_rule_id,
        "subspace_planner": identity.subspace_planner_id,
        "sync_module": identity.sync_module_id
    }
    impl_object_by_domain = {
        "content_extractor": impl_set.content_extractor,
        "geometry_extractor": impl_set.geometry_extractor,
        "fusion_rule": impl_set.fusion_rule,
        "subspace_planner": impl_set.subspace_planner,
        "sync_module": impl_set.sync_module
    }

    for domain, field_path in field_paths_by_domain.items():
        impl_id = impl_id_by_domain.get(domain)
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_id for domain={domain}, field_path={field_path}"
            )
        set_value_by_field_path(record, field_path, impl_id)

    for domain, field_path in version_field_paths_by_domain.items():
        impl_object = impl_object_by_domain.get(domain)
        impl_version = getattr(impl_object, "impl_version", None)
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_version for domain={domain}, field_path={field_path}"
            )
        set_value_by_field_path(record, field_path, impl_version)

    for domain, field_path in digest_field_paths_by_domain.items():
        impl_object = impl_object_by_domain.get(domain)
        impl_digest = getattr(impl_object, "impl_digest", None)
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 缺失或类型不合法，必须 fail-fast。
            raise MissingRequiredFieldError(
                f"Missing required impl_digest for domain={domain}, field_path={field_path}"
            )
        set_value_by_field_path(record, field_path, impl_digest)


def set_failure_status(run_meta: Dict[str, Any], reason: RunFailureReason, exc: Exception) -> None:
    """
    功能：设置失败状态与原因。

    Set failure status with controlled reason and structured details.
    Ensures GateEnforcementError is serialized with structured fields.
    For all other exceptions, provides exc_type, message, and stack_fingerprint.

    Args:
        run_meta: Run metadata mapping to mutate.
        reason: Run failure reason enum.
        exc: Raised exception.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    from main.core.errors import GateEnforcementError
    import sys

    run_meta["status_ok"] = False
    run_meta["status_reason"] = reason

    # 结构化 GateEnforcementError。
    if isinstance(exc, GateEnforcementError):
        run_meta["status_details"] = {
            "gate_name": exc.gate_name if exc.gate_name is not None else "<absent>",
            "field_path": exc.field_path if exc.field_path is not None else "<absent>",
            "expected": exc.expected if exc.expected is not None else "<absent>",
            "actual": exc.actual if exc.actual is not None else "<absent>",
            "message": str(exc)
        }
    else:
        # 其他异常：提取 exc_type、message、stack_fingerprint。
        exc_type = type(exc).__name__
        message = str(exc)

        # 生成稳定的 stack_fingerprint：取首帧的 module:function:lineno。
        stack_fingerprint = "<absent>"
        try:
            tb = sys.exc_info()[2]
            if tb is not None:
                # 获取首帧。
                frame = tb.tb_frame
                filename = frame.f_code.co_filename
                function = frame.f_code.co_name
                lineno = tb.tb_lineno
                # 简化路径为模块名。
                from pathlib import Path
                try:
                    module_path = Path(filename).relative_to(Path.cwd())
                    module_name = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
                except ValueError:
                    module_name = Path(filename).name.replace(".py", "")
                stack_fingerprint = f"{module_name}:{function}:{lineno}"
        except Exception:
            # 若提取失败，保持 <absent>。
            pass

        run_meta["status_details"] = {
            "exc_type": exc_type,
            "message": message,
            "stack_fingerprint": stack_fingerprint
        }


def build_seed_audit(cfg: Any, command: Any) -> Tuple[Dict[str, Any], str, int, str]:
    """
    功能：构造 seed 审计字段与派生 seed_value。

    Build seed audit fields and derived seed value from seed_parts.

    Args:
        cfg: Configuration mapping.
        command: Command name for seed purpose.

    Returns:
        Tuple of (seed_parts, seed_digest, seed_value, seed_rule_id).

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If seed_parts are invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not isinstance(command, str) or not command:
        # command 类型不符合预期，必须 fail-fast。
        raise TypeError("command must be non-empty str")
    cfg_dict = cast(Dict[str, Any], cfg)
    validated_command = command

    seed_parts_cfg = cfg_dict.get("seed_parts")
    if seed_parts_cfg is not None:
        if not isinstance(seed_parts_cfg, dict):
            # seed_parts 类型不符合预期，必须 fail-fast。
            raise TypeError("seed_parts must be dict")
        seed_parts_cfg_dict = cast(Dict[str, Any], seed_parts_cfg)
        actual_keys = set(seed_parts_cfg_dict.keys())
        if actual_keys != _REQUIRED_SEED_PART_KEYS:
            raise ValueError(
                "seed_parts keys mismatch: "
                f"expected={sorted(_REQUIRED_SEED_PART_KEYS)}, actual={sorted(actual_keys)}"
            )
        seed_parts: Dict[str, Any] = dict(seed_parts_cfg_dict)
    else:
        key_id = "<absent>"
        watermark = cfg_dict.get("watermark")
        if isinstance(watermark, dict):
            watermark_dict = cast(Dict[str, Any], watermark)
            key_id_value = watermark_dict.get("key_id")
            if isinstance(key_id_value, str) and key_id_value:
                key_id = key_id_value

        sample_idx = 0
        seed_value = cfg_dict.get("seed")
        if isinstance(seed_value, int):
            sample_idx = seed_value
        elif isinstance(seed_value, str) and seed_value.isdigit():
            sample_idx = int(seed_value)

        seed_parts = {
            "key_id": key_id,
            "sample_idx": sample_idx,
            "purpose": validated_command
        }

    digests.normalize_for_digest(seed_parts)
    seed_digest = digests.canonical_sha256(seed_parts)
    seed_value = time_utils.stable_seed_from_parts(seed_parts)
    return seed_parts, seed_digest, seed_value, _SEED_RULE_ID


def build_determinism_controls(cfg: Any) -> Optional[Dict[str, Any]]:
    """
    功能：构造 determinism_controls 审计字段。

    Build determinism_controls audit mapping from config.

    Args:
        cfg: Configuration mapping.

    Returns:
        Determinism controls mapping or None if empty.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    cfg_dict = cast(Dict[str, Any], cfg)

    controls: Dict[str, Any] = {}
    determinism_controls = cfg_dict.get("determinism_controls")
    if isinstance(determinism_controls, dict):
        controls.update(cast(Dict[str, Any], determinism_controls))

    for key in [
        "torch_deterministic",
        "cudnn_benchmark",
        "deterministic_algorithms",
        "rng_backend"
    ]:
        if key in cfg_dict and key not in controls:
            controls[key] = cfg_dict.get(key)

    if not controls:
        return None
    return controls


def normalize_nondeterminism_notes(value: Any) -> Optional[Any]:
    """
    功能：规范化 nondeterminism_notes 字段。

    Normalize nondeterminism_notes into str or list[str].

    Args:
        value: Raw notes value.

    Returns:
        Normalized notes or None.

    Raises:
        TypeError: If value type is invalid.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            # notes 为空，必须 fail-fast。
            raise TypeError("nondeterminism_notes must be non-empty str")
        return value
    if isinstance(value, list):
        if not value:
            raise TypeError("nondeterminism_notes must be non-empty list")
        value_list = cast(list[Any], value)
        validated_notes: list[str] = []
        for item in value_list:
            if not isinstance(item, str) or not item:
                raise TypeError("nondeterminism_notes items must be non-empty str")
            validated_notes.append(item)
        return validated_notes
    raise TypeError("nondeterminism_notes must be str, list[str], or None")



def format_fact_sources_mismatch(snapshot: Dict[str, Any], bound_fact_sources: Dict[str, Any]) -> str:
    """
    功能：格式化事实源快照不一致错误信息。

    Format snapshot mismatch details for fact sources.

    Args:
        snapshot: Snapshot mapping built before binding context.
        bound_fact_sources: Mapping from records_io.get_bound_fact_sources().

    Returns:
        Mismatch message with field path and both values.
    """
    keys = sorted(set(snapshot.keys()) | set(bound_fact_sources.keys()))
    for key in keys:
        left = snapshot.get(key)
        right = bound_fact_sources.get(key)
        if left != right:
            return (
                "bound_fact_sources snapshot mismatch: "
                f"field={key}, snapshot={left}, bound={right}"
            )

    return "bound_fact_sources snapshot mismatch: field=unknown"
