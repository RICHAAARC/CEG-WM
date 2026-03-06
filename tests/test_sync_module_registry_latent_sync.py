"""
File purpose: 同步模块 registry 与白名单补全回归测试。
Module type: General module
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np

from main.policy.runtime_whitelist import load_runtime_whitelist
from main.registries.geometry_registry import resolve_sync_module
from main.watermarking.geometry_chain.sync import SyncRuntimeContext, resolve_enable_latent_sync
from main.watermarking.detect import orchestrator as detect_orchestrator


def _build_minimal_cfg(enable_latent_sync: bool = True) -> Dict[str, Any]:
    """
    功能：构造最小同步配置。

    Build minimal config mapping for sync module tests.

    Args:
        enable_latent_sync: Whether latent sync is enabled.

    Returns:
        Minimal config mapping.

    Raises:
        TypeError: If enable_latent_sync is not bool.
    """
    if not isinstance(enable_latent_sync, bool):
        # enable_latent_sync 类型不合法，必须 fail-fast。
        raise TypeError("enable_latent_sync must be bool")
    return {
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "detect": {
            "geometry": {
                "enabled": bool(enable_latent_sync),
                "enable_latent_sync": bool(enable_latent_sync),
            }
        }
    }


def test_sync_module_registry_contains_geometry_latent_sync_sd3_v1() -> None:
    """
    功能：sync module registry 必须包含 geometry_latent_sync_sd3_v1。

    Sync module registry must resolve geometry_latent_sync_sd3_v1.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_sync_module("geometry_latent_sync_sd3_v1")
    instance = factory({})
    assert hasattr(instance, "sync")


def test_runtime_whitelist_allows_geometry_latent_sync_sd3_v1() -> None:
    """
    功能：runtime whitelist 必须允许 geometry_latent_sync_sd3_v1。

    Runtime whitelist must allow geometry_latent_sync_sd3_v1 for sync_module domain.

    Args:
        None.

    Returns:
        None.
    """
    whitelist = load_runtime_whitelist()
    impl_cfg = whitelist.data.get("impl_id", {})
    allowed_by_domain = impl_cfg.get("allowed_by_domain", {})
    allowed_sync = allowed_by_domain.get("sync_module", [])
    assert "geometry_latent_sync_sd3_v1" in allowed_sync


def test_sync_module_sync_returns_structured_status() -> None:
    """
    功能：sync_module.sync 必须返回结构化状态字段。

    sync_module.sync must return structured status fields.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_sync_module("geometry_latent_sync_sd3_v1")
    instance = factory({})
    cfg = _build_minimal_cfg(enable_latent_sync=True)
    result = instance.sync(cfg)

    assert isinstance(result, dict)
    assert "sync_status" in result
    assert "sync_success" in result
    assert "impl_identity" in result
    assert "impl_version" in result
    assert "impl_digest" in result

    sync_status = result.get("sync_status")
    assert sync_status in {"ok", "absent", "mismatch", "fail"}
    assert isinstance(result.get("sync_success"), bool)
    if sync_status != "ok":
        assert result.get("sync_success") is False


def test_sync_module_sync_with_context_returns_status() -> None:
    """
    功能：sync_module.sync_with_context 必须返回结构化状态字段。

    sync_module.sync_with_context must return structured status fields.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_sync_module("geometry_latent_sync_sd3_v1")
    instance = factory({})
    cfg = _build_minimal_cfg(enable_latent_sync=True)
    context = SyncRuntimeContext(pipeline=None, latents=None, rng=None, trajectory_evidence=None)
    result = instance.sync_with_context(cfg, context)

    assert isinstance(result, dict)
    assert "sync_status" in result
    assert "sync_success" in result
    sync_status = result.get("sync_status")
    assert sync_status in {"ok", "absent", "mismatch", "fail"}


def test_resolve_enable_latent_sync_flag() -> None:
    """
    功能：resolve_enable_latent_sync 必须遵循配置开关。

    resolve_enable_latent_sync must follow config flags.

    Args:
        None.

    Returns:
        None.
    """
    cfg_enabled = _build_minimal_cfg(enable_latent_sync=True)
    cfg_disabled = _build_minimal_cfg(enable_latent_sync=False)
    assert resolve_enable_latent_sync(cfg_enabled) is True
    assert resolve_enable_latent_sync(cfg_disabled) is False


def test_sync_module_v2_sync_with_context_ok_path() -> None:
    """
    功能：v2 sync_with_context 在完整 runtime_inputs 下应稳定返回非异常状态。

    v2 sync_with_context should return structured status without NameError
    when runtime_inputs includes relation_digest.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_sync_module("geometry_latent_sync_sd3_v2")
    instance = factory({})
    cfg = _build_minimal_cfg(enable_latent_sync=True)
    latents = np.random.default_rng(20260227).normal(size=(1, 4, 8, 8)).astype(np.float32)
    context = SyncRuntimeContext(
        pipeline=object(),
        latents=latents,
        rng=None,
        trajectory_evidence=None,
    )
    result = instance.sync_with_context(
        cfg,
        context,
        runtime_inputs={"relation_digest": "r" * 64},
    )
    assert isinstance(result, dict)
    assert result.get("sync_status") in {"ok", "absent", "mismatch", "failed"}
    assert isinstance(result.get("sync_success"), bool)
    failure_reason = result.get("geometry_failure_reason")
    if isinstance(failure_reason, str):
        assert "NameError" not in failure_reason


def test_sync_module_v2_relation_digest_missing_returns_mismatch() -> None:
    """
    功能：v2 在缺失 relation_digest 时必须返回 absent 语义（embed 侧无 anchor，属正常路径）。

    v2 must return absent semantics when relation_digest is missing,
    reflecting the embed-side execution path where no anchor is available.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_sync_module("geometry_latent_sync_sd3_v2")
    instance = factory({})
    cfg = _build_minimal_cfg(enable_latent_sync=True)
    latents = np.random.default_rng(20260228).normal(size=(1, 4, 8, 8)).astype(np.float32)
    context = SyncRuntimeContext(
        pipeline=object(),
        latents=latents,
        rng=None,
        trajectory_evidence=None,
    )
    result = instance.sync_with_context(cfg, context)
    assert isinstance(result, dict)
    assert result.get("sync_status") == "absent"
    assert result.get("sync_success") is False
    assert result.get("geometry_absent_reason") == "relation_digest_absent_embed_mode"


def test_detect_run_sync_module_for_v2_no_nameerror() -> None:
    """
    功能：detect 侧调用 v2 sync 模块不应出现 sync_with_context_failed: NameError。

    detect orchestrator should not emit sync_with_context_failed: NameError
    for v2 sync module.

    Args:
        None.

    Returns:
        None.
    """
    factory = resolve_sync_module("geometry_latent_sync_sd3_v2")
    instance = factory({})
    cfg = _build_minimal_cfg(enable_latent_sync=True)
    latents = np.random.default_rng(20260301).normal(size=(1, 4, 8, 8)).astype(np.float32)
    runtime_inputs = {
        "pipeline": object(),
        "latents": latents,
        "rng": None,
        "relation_digest": "r" * 64,
    }
    run_sync = getattr(detect_orchestrator, "_run_sync_module_for_detect")
    result = run_sync(instance, cfg, runtime_inputs)
    assert isinstance(result, dict)
    failure_reason = result.get("geometry_failure_reason")
    if isinstance(failure_reason, str):
        assert "sync_with_context_failed: NameError" not in failure_reason
