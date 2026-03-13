"""
CLI 入口层公共辅助逻辑

功能说明：
- 提供 CLI 入口层共享的冻结面相关辅助函数，避免入口分叉导致的语义漂移。
"""

import os
from typing import Dict, Any, Optional, Tuple
from main.core.contracts import FrozenContracts, get_contract_interpretation
from main.core.errors import MissingRequiredFieldError, RunFailureReason
from main.registries import runtime_resolver
from main.core import digests
from main.diffusion.sd3.callback_composer import InjectionContext
from main.watermarking.content_chain import channel_lf, channel_hf
from main.core import time_utils


_SEED_RULE_ID = "stable_seed_from_parts_v1"
_REQUIRED_SEED_PART_KEYS = {"key_id", "sample_idx", "purpose"}


def resolve_attestation_env_inputs(
    cfg: Dict[str, Any],
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

    attestation_node = cfg.get("attestation")
    attestation_cfg = attestation_node if isinstance(attestation_node, dict) else {}
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
    if isinstance(k_master, str) and k_master.strip():
        result["k_master"] = k_master.strip()

    if require_prompt_seed:
        k_prompt = os.environ.get(k_prompt_env_var, "")
        k_seed = os.environ.get(k_seed_env_var, "")
        if isinstance(k_prompt, str) and k_prompt.strip():
            result["k_prompt"] = k_prompt.strip()
        if isinstance(k_seed, str) and k_seed.strip():
            result["k_seed"] = k_seed.strip()

    missing_secret_fields = []
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


def build_cli_config_migration_hint(exc: Exception) -> Optional[str]:
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
    cfg: Dict[str, Any],
    plan_payload: Dict[str, Any],
    plan_digest: str
) -> InjectionContext:
    """
    功能：基于 plan 构造 InjectionContext（不修改 cfg）。
    
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

    watermark_cfg = cfg.get("watermark", {}) if isinstance(cfg.get("watermark", {}), dict) else {}
    lf_cfg = watermark_cfg.get("lf", {}) if isinstance(watermark_cfg.get("lf", {}), dict) else {}
    hf_cfg = watermark_cfg.get("hf", {}) if isinstance(watermark_cfg.get("hf", {}), dict) else {}

    enable_lf = bool(lf_cfg.get("enabled", False))
    enable_hf = bool(hf_cfg.get("enabled", False))

    lf_strength = lf_cfg.get("strength", cfg.get("lf_strength", 1.5))
    hf_threshold_percentile = hf_cfg.get("threshold_percentile", cfg.get("hf_threshold_percentile", 75.0))

    lf_params = {
        "impl_id": channel_lf.LF_CHANNEL_IMPL_ID,
        "impl_version": channel_lf.LF_CHANNEL_VERSION,
        "lf_strength": lf_strength,
        "lf_enabled": enable_lf
    }
    hf_params = {
        "impl_id": channel_hf.HF_CHANNEL_IMPL_ID,
        "impl_version": channel_hf.HF_CHANNEL_VERSION,
        "hf_threshold_percentile": hf_threshold_percentile,
        "hf_enabled": enable_hf
    }
    lf_params_digest = digests.canonical_sha256(lf_params) if enable_lf else ""
    hf_params_digest = digests.canonical_sha256(hf_params) if enable_hf else ""

    device = cfg.get("device", "cpu")
    dtype = cfg.get("dtype", "float32")
    if isinstance(cfg.get("model"), dict) and "dtype" in cfg.get("model"):
        dtype = cfg.get("model").get("dtype", dtype)

    return InjectionContext(
        plan_digest=plan_digest,
        plan_ref=plan_payload,
        lf_params_digest=lf_params_digest,
        hf_params_digest=hf_params_digest,
        enable_lf=enable_lf,
        enable_hf=enable_hf,
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
    功能：设置失败状态与原因（结构化）。

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
    import traceback
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
                # 获取首帧（抛出点）。
                frame = tb.tb_frame
                filename = frame.f_code.co_filename
                function = frame.f_code.co_name
                lineno = tb.tb_lineno
                # 简化路径为模块名（去掉绝对路径前缀）。
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


def build_seed_audit(cfg: Dict[str, Any], command: str) -> Tuple[Dict[str, Any], str, int, str]:
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

    seed_parts_cfg = cfg.get("seed_parts")
    if seed_parts_cfg is not None:
        if not isinstance(seed_parts_cfg, dict):
            # seed_parts 类型不符合预期，必须 fail-fast。
            raise TypeError("seed_parts must be dict")
        if set(seed_parts_cfg.keys()) != _REQUIRED_SEED_PART_KEYS:
            raise ValueError(
                "seed_parts keys mismatch: "
                f"expected={sorted(_REQUIRED_SEED_PART_KEYS)}, actual={sorted(seed_parts_cfg.keys())}"
            )
        seed_parts = dict(seed_parts_cfg)
    else:
        key_id = "<absent>"
        watermark = cfg.get("watermark")
        if isinstance(watermark, dict):
            key_id_value = watermark.get("key_id")
            if isinstance(key_id_value, str) and key_id_value:
                key_id = key_id_value

        sample_idx = 0
        seed_value = cfg.get("seed")
        if isinstance(seed_value, int):
            sample_idx = seed_value
        elif isinstance(seed_value, str) and seed_value.isdigit():
            sample_idx = int(seed_value)

        seed_parts = {
            "key_id": key_id,
            "sample_idx": sample_idx,
            "purpose": command
        }

    digests.normalize_for_digest(seed_parts)
    seed_digest = digests.canonical_sha256(seed_parts)
    seed_value = time_utils.stable_seed_from_parts(seed_parts)
    return seed_parts, seed_digest, seed_value, _SEED_RULE_ID


def build_determinism_controls(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

    controls = {}
    if isinstance(cfg.get("determinism_controls"), dict):
        controls.update(cfg.get("determinism_controls"))

    for key in [
        "torch_deterministic",
        "cudnn_benchmark",
        "deterministic_algorithms",
        "rng_backend"
    ]:
        if key in cfg and key not in controls:
            controls[key] = cfg.get(key)

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
        for item in value:
            if not isinstance(item, str) or not item:
                raise TypeError("nondeterminism_notes items must be non-empty str")
        return list(value)
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
