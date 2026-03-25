"""
File purpose: 注入回调在 infer_runtime 中生效与降级测试。
Module type: General module

功能说明：
- 验证 infer_runtime 中注入回调能产生注入证据摘要。
- 验证不支持 callback 的 pipeline 会稳定降级为 absent。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch

from main.core import digests
from main.diffusion.sd3.callback_composer import InjectionContext
from main.diffusion.sd3.infer_runtime import run_sd3_inference
from main.diffusion.sd3 import infer_runtime as infer_runtime_module
from main.cli.run_common import build_injection_context_from_plan
from main.watermarking.content_chain import channel_lf
from main.watermarking.content_chain.latent_modifier import (
    LatentModifier,
    LATENT_MODIFIER_ID,
    LATENT_MODIFIER_VERSION
)


class _CallbackPipelineStub:
    """
    功能：支持 callback_on_step_end 的推理桩。

    Callback-capable pipeline stub for injection evidence.
    """

    def __init__(self, base_seed: int) -> None:
        self._base_seed = base_seed

    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        callback_on_step_end: Any = None,
        callback_on_step_end_tensor_inputs: Any = None
    ) -> Any:
        if callback_on_step_end is not None:
            for step_index in range(num_inference_steps):
                rng = np.random.default_rng(self._base_seed + step_index)
                latents = rng.normal(0.0, 1.0, size=(1, 4, 8, 8)).astype(np.float32)
                callback_on_step_end(
                    self,
                    step_index,
                    step_index,
                    {"latents": latents}
                )
        class _Output:
            images = [object()]
        return _Output()


class _NoCallbackPipelineStub:
    """
    功能：不支持 callback 的推理桩。

    Pipeline stub without callback support.
    """

    def __call__(self, **kwargs: Any) -> Any:
        class _Output:
            images = [object()]
        return _Output()


def _build_cfg() -> Dict[str, Any]:
    return {
        "inference_enabled": True,
        "inference_prompt": "prompt",
        "inference_num_steps": 3,
        "inference_guidance_scale": 7.0,
        "inference_height": 512,
        "inference_width": 512,
        "trajectory_tap": {"enabled": True},
        "watermark": {
            "lf": {"enabled": True, "strength": 0.5},
            "hf": {"enabled": True, "threshold_percentile": 75.0},
            "subspace": {
                "sample_count": 3,
                "feature_dim": 16,
                "timestep_start": 0,
                "timestep_end": 2
            }
        },
        "attestation": {"use_trajectory_mix": False},
        "attestation_digest": "a" * 64,
        "attestation_event_digest": "b" * 64,
        "lf_attestation_event_digest": "b" * 64,
        "lf_attestation_key": "c" * 64,
        "k_lf": "c" * 64,
        "event_binding_mode": "statement_only",
    }


def _stable_qr_basis(matrix: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(matrix, dtype=np.float64), dtype=torch.float64)
    q_tensor, _ = torch.linalg.qr(tensor, mode="reduced")
    return q_tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _build_plan_with_basis(latent_dim: int, rank: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    lf_matrix = rng.normal(0.0, 1.0, size=(latent_dim, rank)).astype(np.float32)
    lf_matrix = _stable_qr_basis(lf_matrix)
    hf_matrix = rng.normal(0.0, 1.0, size=(latent_dim, rank)).astype(np.float32)
    hf_matrix = _stable_qr_basis(hf_matrix)
    return {
        "basis_digest": digests.canonical_sha256({"seed": seed, "rank": rank, "tag": "basis"}),
        "lf_basis": {
            "projection_matrix": lf_matrix.tolist(),
            "trajectory_feature_spec": {
                "feature_operator": "masked_normalized_random_projection",
                "edit_timestep": 1,
            },
        },
        "hf_basis": {"hf_projection_matrix": hf_matrix.tolist()},
        "planner_params": {"rank": rank}
    }


_build_injection_cfg = infer_runtime_module._build_injection_cfg  # pyright: ignore[reportPrivateUsage]
extract_image_conditioned_latent = infer_runtime_module.extract_image_conditioned_latent


class _VaeLatentDistributionStub:
    def __init__(self, latents: torch.Tensor) -> None:
        self.mean = latents


class _VaeEncodeOutputStub:
    def __init__(self, latents: torch.Tensor) -> None:
        self.latent_dist = _VaeLatentDistributionStub(latents)


class _VaeStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.config = type("_VaeConfig", (), {"scaling_factor": 0.5})()

    def encode(self, tensor: torch.Tensor) -> _VaeEncodeOutputStub:
        latents = torch.mean(tensor, dim=1, keepdim=True)[:, :, ::8, ::8]
        return _VaeEncodeOutputStub(latents)


class _PipelineWithVaeStub:
    def __init__(self) -> None:
        self.vae = _VaeStub()


def test_injection_callback_smoke() -> None:
    """
    功能：注入回调必须产生可复算证据摘要。
    """
    cfg = _build_cfg()
    latent_dim = 1 * 4 * 8 * 8
    plan_payload = _build_plan_with_basis(latent_dim, 8, 2026)
    plan_digest = digests.canonical_sha256(plan_payload)

    injection_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)
    modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)

    result = run_sd3_inference(
        cfg,
        _CallbackPipelineStub(base_seed=123),
        "cpu",
        None,
        injection_context=injection_context,
        injection_modifier=modifier
    )

    injection_evidence = result.get("injection_evidence")
    assert isinstance(injection_evidence, dict)
    assert injection_evidence.get("status") == "ok"
    assert isinstance(injection_evidence.get("injection_trace_digest"), str)
    assert len(injection_evidence.get("injection_trace_digest")) == 64
    assert isinstance(injection_evidence.get("injection_params_digest"), str)
    assert len(injection_evidence.get("injection_params_digest")) == 64

    metrics = injection_evidence.get("injection_metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("step_count") == 3
    assert isinstance(metrics.get("delta_norm_mean"), float)
    assert metrics.get("delta_norm_mean") > 0.0
    assert isinstance(metrics.get("lf_delta_norm_mean"), float)
    assert isinstance(metrics.get("hf_delta_norm_mean"), float)
    assert metrics.get("lf_edit_timestep") == 1
    assert metrics.get("lf_closed_loop_step_index") == 2
    assert metrics.get("lf_edit_timestep_step_index") == 1
    assert metrics.get("lf_terminal_step_index") == 2
    assert isinstance(metrics.get("lf_edit_timestep_closed_loop_summary"), dict)
    assert isinstance(metrics.get("lf_terminal_step_closed_loop_summary"), dict)


def test_injection_unsupported_callback_absent() -> None:
    """
    功能：不支持 callback 的 pipeline 必须稳定降级为 absent。
    """
    cfg = _build_cfg()
    latent_dim = 1 * 4 * 8 * 8
    plan_payload = _build_plan_with_basis(latent_dim, 8, 2026)
    plan_digest = digests.canonical_sha256(plan_payload)

    injection_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)
    modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)

    result = run_sd3_inference(
        cfg,
        _NoCallbackPipelineStub(),
        "cpu",
        None,
        injection_context=injection_context,
        injection_modifier=modifier
    )

    injection_evidence = result.get("injection_evidence")
    assert isinstance(injection_evidence, dict)
    assert injection_evidence.get("status") == "absent"
    assert injection_evidence.get("injection_absent_reason") == "unsupported_pipeline"


def test_injection_params_not_in_digest_domain_must_mismatch() -> None:
    """
    功能：注入参数摘要不一致必须触发 mismatch。
    """
    cfg = _build_cfg()
    latent_dim = 1 * 4 * 8 * 8
    plan_payload = _build_plan_with_basis(latent_dim, 8, 2026)
    plan_digest = digests.canonical_sha256(plan_payload)

    valid_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)
    invalid_context = InjectionContext(
        plan_digest=valid_context.plan_digest,
        plan_ref=valid_context.plan_ref,
        lf_params_digest="0" * 64,
        hf_params_digest=valid_context.hf_params_digest,
        enable_lf=valid_context.enable_lf,
        enable_hf=valid_context.enable_hf,
        device=valid_context.device,
        dtype=valid_context.dtype,
    )

    modifier = LatentModifier(LATENT_MODIFIER_ID, LATENT_MODIFIER_VERSION)
    result = run_sd3_inference(
        cfg,
        _CallbackPipelineStub(base_seed=123),
        "cpu",
        None,
        injection_context=invalid_context,
        injection_modifier=modifier,
    )

    injection_evidence = result.get("injection_evidence")
    assert isinstance(injection_evidence, dict)
    assert injection_evidence.get("status") == "mismatch"
    assert injection_evidence.get("injection_failure_reason") == "lf_params_digest_mismatch"


def test_build_injection_cfg_carries_attestation_runtime_fields() -> None:
    """
    功能：逐 step 注入配置必须显式携带 LF formal sign 所需锚点。
    """
    cfg = _build_cfg()
    latent_dim = 1 * 4 * 8 * 8
    plan_payload = _build_plan_with_basis(latent_dim, 8, 2026)
    plan_digest = digests.canonical_sha256(plan_payload)

    injection_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)
    injection_cfg = _build_injection_cfg(cfg, injection_context)

    assert injection_cfg.get("attestation_event_digest") == "b" * 64
    assert injection_cfg.get("lf_attestation_event_digest") == "b" * 64
    assert injection_cfg.get("lf_attestation_key") == "c" * 64
    assert injection_cfg.get("k_lf") == "c" * 64
    assert injection_cfg.get("basis_digest") == plan_payload["basis_digest"]
    assert injection_cfg.get("lf_basis_digest") == plan_payload["basis_digest"]
    assert injection_cfg.get("event_binding_mode") == "statement_only"


def test_build_injection_context_resolves_statement_only_from_attestation_cfg() -> None:
    """
    功能：early runtime 缺失时，InjectionContext 仍必须从 attestation 配置解析 statement_only。
    """
    cfg = _build_cfg()
    cfg.pop("event_binding_mode", None)
    cfg.pop("attestation_digest", None)
    cfg.pop("attestation_event_digest", None)
    cfg.pop("lf_attestation_event_digest", None)
    cfg.pop("lf_attestation_key", None)
    cfg.pop("k_lf", None)

    latent_dim = 1 * 4 * 8 * 8
    plan_payload = _build_plan_with_basis(latent_dim, 8, 2026)
    plan_digest = digests.canonical_sha256(plan_payload)

    injection_context = build_injection_context_from_plan(cfg, plan_payload, plan_digest)

    assert injection_context.event_binding_mode == "statement_only"


def test_channel_lf_resolves_statement_only_from_attestation_cfg() -> None:
    """
    功能：LF 通道在 runtime 字段缺失时仍必须遵循 attestation.use_trajectory_mix 的规范化语义。
    """
    cfg = {
        "attestation": {"use_trajectory_mix": False},
    }

    assert channel_lf._resolve_event_binding_mode(cfg) == "statement_only"


def test_extract_image_conditioned_latent_returns_object_bound_latent(tmp_path: Path) -> None:
    """
    功能：detect 输入图像必须可通过 VAE encode 恢复 object-bound latent。
    """
    image_path = tmp_path / "watermarked.png"
    image_array = np.full((16, 16, 3), 128, dtype=np.uint8)
    Image.fromarray(image_array, mode="RGB").save(image_path)

    cfg = _build_cfg()
    cfg["inference_height"] = 16
    cfg["inference_width"] = 16

    result = extract_image_conditioned_latent(
        cfg,
        _PipelineWithVaeStub(),
        str(image_path),
        "cpu",
    )

    assert result.get("status") == "ok"
    latent_array = result.get("latent_array")
    assert isinstance(latent_array, np.ndarray)
    assert latent_array.shape == (1, 1, 2, 2)
    assert result.get("latent_source") == "input_image_vae_encode"
    assert result.get("image_size") == [16, 16]
