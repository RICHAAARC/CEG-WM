"""
统一 latent 修改器（水印注入统一入口）

功能说明：
- 统一调度 LF/HF 子空间修改。
- 在 SD3 推理循环中修改 latents 张量。
- 生成完整的 channel 侧证据与审计信息。
- 严格遵循：仅改 latent，不写原始张量到 records。

Module type: Core innovation module
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from main.core import digests
from main.watermarking.content_chain import channel_lf
from main.watermarking.content_chain import channel_hf


LATENT_MODIFIER_ID = "unified_latent_modifier_v1"
LATENT_MODIFIER_VERSION = "v1"


class LatentModifier:
    """
    功能：统一 latent 修改器。
    
    Unified latent modifier orchestrating LF/HF channels.
    Coordinates subspace-based watermark injection during diffusion inference.
    
    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version.
    
    Returns:
        None.
    
    Raises:
        ValueError: If constructor inputs are invalid.
    """
    
    def __init__(self, impl_id: str, impl_version: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            raise ValueError("impl_version must be non-empty str")
        self.impl_id = impl_id
        self.impl_version = impl_version
    
    def apply_latent_update(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        step_index: Optional[int] = None,
        key: Optional[int] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        功能：在单个推理 step 中修改 latents 并产出证据。
        
        Apply watermark modifications to latents during inference step.
        Orchestrates LF and HF channel updates, returns modified latents and step evidence.
        
        Args:
            latents: Input latent tensor (torch.Tensor or np.ndarray).
            plan: Subspace plan mapping from SubspacePlanner (contains basis info).
            cfg: Configuration mapping with channel parameters.
            step_index: Current diffusion step index (for tracing).
            key: Random seed for encoding (derived from cfg + step if not provided).
        
        Returns:
            Tuple of (latents_modified, step_evidence_dict).
            step_evidence_dict contains channel-wise traces and metrics (no raw tensors).
        
        Raises:
            TypeError: If inputs types are invalid.
            ValueError: If plan or cfg parameters invalid.
        """
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be dict")
        
        if plan is not None and not isinstance(plan, dict):
            raise TypeError("plan must be dict or None")
        
        # 支持 torch.Tensor 转换为 np 数组。
        is_torch_input = hasattr(latents, "cpu")
        if is_torch_input:
            latents_np = latents.cpu().numpy().copy()
        else:
            latents_np = np.asarray(latents, dtype=np.float32).copy()
        
        original_shape = latents_np.shape
        latents_working = latents_np.copy()
        
        # 记录初始状态。
        latents_before_modifications = latents_working.copy()
        
        # 提取基础参数。
        lf_enabled = cfg.get("lf_enabled", True)
        hf_enabled = cfg.get("hf_enabled", True)
        
        if key is None:
            # 衍生 key：基于 step_index 与固定种子。
            seed_base = cfg.get("watermark_seed", 42)
            if step_index is not None:
                key = seed_base + step_index
            else:
                key = seed_base
        
        step_evidence = {
            "step_index": step_index if step_index is not None else -1,
            "latents_shape": list(original_shape),
            "lf_evidence": None,
            "hf_evidence": None,
            "combined_status": "ok"
        }
        
        try:
            # ============ LF 通道处理 ============
            if lf_enabled and plan is not None:
                try:
                    basis_lf = plan.get("lf_basis")
                    if basis_lf is None:
                        lf_evidence = {
                            "status": "absent",
                            "absent_reason": "lf_basis_missing_in_plan",
                            "encoded_coeffs": None
                        }
                    else:
                        # 投影到 LF 子空间。
                        lf_coeffs = channel_lf.compute_lf_basis_projection(
                            latents_working, basis_lf
                        )
                        
                        # 施加 LF 编码。
                        lf_coeffs_encoded, encoding_evidence = channel_lf.apply_low_freq_encoding(
                            lf_coeffs, key, cfg
                        )
                        
                        # 从编码系数恢复修改后的 latents。
                        latents_lf_modified = channel_lf.reconstruct_from_lf_coeffs(
                            lf_coeffs_encoded, basis_lf, original_shape, is_torch=False
                        )
                        
                        # 对全局 latents 应用 LF 修改。
                        # 采用混合策略：保留原始的高频部分，
                        # 将 LF 修改与原始 latents 的 HF 部分结合。
                        latents_lf_delta = latents_lf_modified - channel_lf.reconstruct_from_lf_coeffs(
                            lf_coeffs, basis_lf, original_shape, is_torch=False
                        )
                        latents_working = latents_working + latents_lf_delta
                        
                        # 构造 step 证据（含摘要但不含原始张量）。
                        lf_evidence = {
                            "status": "ok",
                            "coeffs_norm": float(np.linalg.norm(lf_coeffs)),
                            "encoded_coeffs_norm": float(np.linalg.norm(lf_coeffs_encoded)),
                            "encoding_evidence": encoding_evidence
                        }
                except Exception as e:
                    lf_evidence = {
                        "status": "failed",
                        "failure_reason": f"lf_encoding_error: {type(e).__name__}",
                        "error_detail": str(e)
                    }
                    step_evidence["combined_status"] = "failed"
            else:
                lf_evidence = {
                    "status": "absent",
                    "absent_reason": "lf_disabled_by_config" if not lf_enabled else "lf_plan_missing"
                }
            
            step_evidence["lf_evidence"] = lf_evidence
            
            # ============ HF 通道处理 ============
            if hf_enabled and plan is not None:
                try:
                    basis_hf = plan.get("hf_basis")
                    if basis_hf is None:
                        hf_evidence = {
                            "status": "absent",
                            "absent_reason": "hf_basis_missing_in_plan"
                        }
                    else:
                        # 投影到 HF 子空间。
                        hf_coeffs = channel_hf.compute_hf_basis_projection(
                            latents_working, basis_hf
                        )
                        
                        # 施加 HF 约束。
                        hf_coeffs_constrained, constraint_evidence = channel_hf.apply_hf_truncation_constraint(
                            hf_coeffs, cfg
                        )
                        
                        # 从约束系数恢复修改后的 latents。
                        latents_hf_modified = channel_hf.reconstruct_from_hf_coeffs(
                            hf_coeffs_constrained, basis_hf, original_shape, is_torch=False
                        )
                        
                        # 对全局 latents 应用 HF 修改。
                        latents_hf_delta = latents_hf_modified - channel_hf.reconstruct_from_hf_coeffs(
                            hf_coeffs, basis_hf, original_shape, is_torch=False
                        )
                        latents_working = latents_working + latents_hf_delta
                        
                        hf_evidence = {
                            "status": "ok",
                            "coeffs_norm": float(np.linalg.norm(hf_coeffs)),
                            "constrained_coeffs_norm": float(np.linalg.norm(hf_coeffs_constrained)),
                            "constraint_evidence": constraint_evidence
                        }
                except Exception as e:
                    hf_evidence = {
                        "status": "failed",
                        "failure_reason": f"hf_constraint_error: {type(e).__name__}",
                        "error_detail": str(e)
                    }
                    if step_evidence["combined_status"] == "ok":
                        step_evidence["combined_status"] = "failed"
            else:
                hf_evidence = {
                    "status": "absent",
                    "absent_reason": "hf_disabled_by_config" if not hf_enabled else "hf_plan_missing"
                }
            
            step_evidence["hf_evidence"] = hf_evidence
            
            # 记录修改后的状态。
            step_evidence["latents_before_norm"] = float(np.linalg.norm(latents_before_modifications))
            step_evidence["latents_after_norm"] = float(np.linalg.norm(latents_working))
            step_evidence["modification_delta_norm"] = float(np.linalg.norm(
                latents_working - latents_before_modifications
            ))
            
        except Exception as e:
            step_evidence["combined_status"] = "failed"
            step_evidence["overall_error"] = str(e)
        
        # 恢复为原始类型（torch 或 numpy）。
        if is_torch_input:
            import torch
            latents_modified = torch.from_numpy(latents_working.astype(np.float32))
        else:
            latents_modified = latents_working.astype(np.float32)
        
        return latents_modified, step_evidence
