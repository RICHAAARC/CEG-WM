"""
SD3 Pipeline Inspector

功能：
- 从 diffusers 的 SD3 pipeline 提取结构指纹（structure fingerprint）。
- 支持 Transformer/DiT 的关键模块检查（不是 UNet）。
- 生成可复算的 pipeline_fingerprint 与 fingerprint_digest。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from main.core import digests


IMPL_ID = "sd3_pipeline_inspector_v1"
IMPL_VERSION = "v1"


def inspect_sd3_pipeline(
    pipeline_obj: Any,
    cfg: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    功能：从 SD3 pipeline 对象提取结构指纹。

    Inspect SD3 pipeline and extract structure fingerprint.
    Supports Transformer/DiT architecture (not UNet).

    Args:
        pipeline_obj: diffusers SD3 pipeline object.
        cfg: Optional configuration mapping for context.

    Returns:
        Tuple of (pipeline_fingerprint dict, fingerprint_digest str).
        p eline_fingerprint 包含关键结构参数。
        fingerprint_digest 是 pipeline_fingerprint 的 canonical sha256。

    Raises:
        TypeError: If pipeline_obj is None or invalid.
        ValueError: If pipeline structure extraction failed.
    """
    if pipeline_obj is None:
        raise TypeError("pipeline_obj must not be None")

    fingerprint = {}

    # 1) Transformer 关键子模块计数与名称摘要。
    transformer = getattr(pipeline_obj, "transformer", None)
    if transformer is not None:
        # Transformer 的关键配置提取。
        transformer_config = getattr(transformer, "config", None)
        if transformer_config is not None:
            # 提取 DiT/Transformer 关键参数。
            fingerprint["transformer_num_blocks"] = _safe_get_attr(
                transformer_config, "num_layers", default="<absent>"
            )
            fingerprint["transformer_attention_head_dim"] = _safe_get_attr(
                transformer_config, "attention_head_dim", default="<absent>"
            )
            fingerprint["transformer_num_attention_heads"] = _safe_get_attr(
                transformer_config, "num_attention_heads", default="<absent>"
            )
            fingerprint["transformer_in_channels"] = _safe_get_attr(
                transformer_config, "in_channels", default="<absent>"
            )
            fingerprint["transformer_out_channels"] = _safe_get_attr(
                transformer_config, "out_channels", default="<absent>"
            )
            fingerprint["transformer_sample_size"] = _safe_get_attr(
                transformer_config, "sample_size", default="<absent>"
            )
            fingerprint["transformer_patch_size"] = _safe_get_attr(
                transformer_config, "patch_size", default="<absent>"
            )
        else:
            fingerprint["transformer_config_status"] = "config_missing"
    else:
        fingerprint["transformer_status"] = "transformer_missing"

    # 2) Scheduler 名称与关键超参。
    scheduler = getattr(pipeline_obj, "scheduler", None)
    if scheduler is not None:
        scheduler_config = getattr(scheduler, "config", None)
        if scheduler_config is not None:
            fingerprint["scheduler_class_name"] = scheduler.__class__.__name__
            fingerprint["scheduler_num_train_timesteps"] = _safe_get_attr(
                scheduler_config, "num_train_timesteps", default="<absent>"
            )
            fingerprint["scheduler_beta_schedule"] = _safe_get_attr(
                scheduler_config, "beta_schedule", default="<absent>"
            )
            fingerprint["scheduler_prediction_type"] = _safe_get_attr(
                scheduler_config, "prediction_type", default="<absent>"
            )
        else:
            fingerprint["scheduler_config_status"] = "config_missing"
    else:
        fingerprint["scheduler_status"] = "scheduler_missing"

    # 3) VAE latent channels。
    vae = getattr(pipeline_obj, "vae", None)
    if vae is not None:
        vae_config = getattr(vae, "config", None)
        if vae_config is not None:
            fingerprint["vae_latent_channels"] = _safe_get_attr(
                vae_config, "latent_channels", default="<absent>"
            )
            fingerprint["vae_in_channels"] = _safe_get_attr(
                vae_config, "in_channels", default="<absent>"
            )
            fingerprint["vae_out_channels"] = _safe_get_attr(
                vae_config, "out_channels", default="<absent>"
            )
        else:
            fingerprint["vae_config_status"] = "config_missing"
    else:
        fingerprint["vae_status"] = "vae_missing"

    # 4) Text encoders/CLIP/T5（若使用）版本摘要。
    text_encoder_names = []
    for encoder_attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
        encoder = getattr(pipeline_obj, encoder_attr, None)
        if encoder is not None:
            encoder_class_name = encoder.__class__.__name__
            text_encoder_names.append(encoder_class_name)
    fingerprint["text_encoder_names"] = text_encoder_names if text_encoder_names else ["<absent>"]

    # 5) 从 cfg 中提取模型版本相关信息（若有）。
    if isinstance(cfg, dict):
        model_id = cfg.get("model_id")
        if isinstance(model_id, str) and model_id:
            fingerprint["model_id"] = model_id
        hf_revision = cfg.get("hf_revision")
        if isinstance(hf_revision, str) and hf_revision:
            fingerprint["hf_revision"] = hf_revision
        resolved_revision = cfg.get("resolved_revision")
        if isinstance(resolved_revision, str) and resolved_revision:
            fingerprint["resolved_revision"] = resolved_revision

    # 6) 生成 fingerprint_digest。
    fingerprint_digest = digests.canonical_sha256(fingerprint)

    return fingerprint, fingerprint_digest


def _safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """
    功能：安全地从对象提取属性。

    Safe attribute extraction with default fallback.

    Args:
        obj: Target object.
        attr: Attribute name.
        default: Fallback value if attribute missing.

    Returns:
        Attribute value or default.
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def get_inspector_impl_identity() -> Dict[str, str]:
    """
    功能：返回 inspector 的实现身份标识。

    Get inspector implementation identity.

    Returns:
        Dict with impl_id, impl_version, impl_digest.
    """
    impl_digest = digests.canonical_sha256({
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "source_module": "main.diffusion.sd3.pipeline_inspector"
    })
    return {
        "impl_id": IMPL_ID,
        "impl_version": IMPL_VERSION,
        "impl_digest": impl_digest
    }
