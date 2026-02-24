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

from typing import Any, Dict, Optional, Tuple, List

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

        lf_enabled = self._resolve_channel_enabled(cfg, "lf")
        hf_enabled = self._resolve_channel_enabled(cfg, "hf")
        require_basis_region_spec = self._resolve_require_basis_region_spec(cfg)
        if require_basis_region_spec:
            self._validate_required_plan_fields(plan, lf_enabled=lf_enabled, hf_enabled=hf_enabled)
        
        is_torch_input = False
        try:
            import torch
            is_torch_input = torch.is_tensor(latents)
        except Exception:
            is_torch_input = False

        if is_torch_input:
            return self._apply_latent_update_torch(
                latents=latents,
                plan=plan,
                cfg=cfg,
                step_index=step_index,
                key=key
            )

        latents_np = np.asarray(latents, dtype=np.float32).copy()
        original_shape = latents_np.shape
        latents_working = latents_np.copy()
        latents_before_modifications = latents_working.copy()
        
        # 提取基础参数。
        
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
                    lf_region_spec = plan.get("lf_region_index_spec")
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
                        latents_lf_delta = self._apply_region_spec_delta_np(latents_lf_delta, lf_region_spec)
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
                    hf_region_spec = plan.get("hf_region_index_spec")
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
                        latents_hf_delta = self._apply_region_spec_delta_np(latents_hf_delta, hf_region_spec)
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

            if (
                isinstance(lf_evidence, dict)
                and isinstance(hf_evidence, dict)
                and lf_evidence.get("status") == "absent"
                and hf_evidence.get("status") == "absent"
                and step_evidence.get("combined_status") == "ok"
            ):
                step_evidence["combined_status"] = "absent"
            
            # 记录修改后的状态。
            step_evidence["latents_before_norm"] = float(np.linalg.norm(latents_before_modifications))
            step_evidence["latents_after_norm"] = float(np.linalg.norm(latents_working))
            step_evidence["modification_delta_norm"] = float(np.linalg.norm(
                latents_working - latents_before_modifications
            ))
            
        except Exception as e:
            step_evidence["combined_status"] = "failed"
            step_evidence["overall_error"] = str(e)
        
        latents_modified = latents_working.astype(np.float32)
        return latents_modified, step_evidence

    def _apply_latent_update_torch(
        self,
        latents: Any,
        plan: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
        step_index: Optional[int],
        key: Optional[int]
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        功能：在 torch 张量上执行真实注入更新。

        Apply LF/HF updates directly on torch latents while preserving
        device and dtype.

        Args:
            latents: Torch tensor latents.
            plan: Subspace plan with LF/HF basis.
            cfg: Injection config mapping.
            step_index: Current diffusion step index.
            key: Optional encoding seed.

        Returns:
            Tuple of (latents_modified, step_evidence).

        Raises:
            TypeError: If torch is unavailable or inputs are invalid.
        """
        import torch

        if not torch.is_tensor(latents):
            raise TypeError("latents must be torch.Tensor")

        original_shape = tuple(latents.shape)
        original_device = latents.device
        original_dtype = latents.dtype
        latents_working = latents
        latents_before_modifications = latents.detach().clone()

        lf_enabled = self._resolve_channel_enabled(cfg, "lf")
        hf_enabled = self._resolve_channel_enabled(cfg, "hf")
        require_basis_region_spec = self._resolve_require_basis_region_spec(cfg)
        if require_basis_region_spec:
            self._validate_required_plan_fields(plan, lf_enabled=lf_enabled, hf_enabled=hf_enabled)

        if key is None:
            seed_base = cfg.get("watermark_seed", 42)
            key = seed_base + int(step_index) if isinstance(step_index, int) else seed_base

        runtime_binding = None
        if isinstance(plan, dict):
            runtime_binding = plan.get("runtime_subspace_binding")
        runtime_binding_digest = None
        if isinstance(runtime_binding, dict):
            digest_value = runtime_binding.get("binding_digest")
            if isinstance(digest_value, str) and digest_value:
                runtime_binding_digest = digest_value

        step_evidence = {
            "step_index": step_index if step_index is not None else -1,
            "latents_shape": list(original_shape),
            "lf_evidence": None,
            "hf_evidence": None,
            "combined_status": "ok",
            "runtime_subspace_binding_digest": runtime_binding_digest,
            "latents_device": str(original_device),
            "latents_dtype": str(original_dtype),
        }

        try:
            lf_delta_norm = 0.0
            hf_delta_norm = 0.0

            if lf_enabled and isinstance(plan, dict):
                try:
                    basis_lf = plan.get("lf_basis")
                    lf_region_spec = plan.get("lf_region_index_spec")
                    if basis_lf is None:
                        lf_evidence = {
                            "status": "absent",
                            "absent_reason": "lf_basis_missing_in_plan",
                            "encoded_coeffs": None
                        }
                    else:
                        lf_coeffs = channel_lf.compute_lf_basis_projection_torch(latents_working, basis_lf)
                        lf_coeffs_encoded, encoding_evidence = channel_lf.apply_low_freq_encoding_torch(
                            lf_coeffs,
                            int(key),
                            cfg
                        )
                        latents_lf_modified = channel_lf.reconstruct_from_lf_coeffs_torch(
                            lf_coeffs_encoded,
                            basis_lf,
                            original_shape,
                            dtype=original_dtype
                        )
                        latents_lf_original = channel_lf.reconstruct_from_lf_coeffs_torch(
                            lf_coeffs,
                            basis_lf,
                            original_shape,
                            dtype=original_dtype
                        )
                        latents_lf_delta = latents_lf_modified - latents_lf_original
                        latents_lf_delta = self._apply_region_spec_delta_torch(latents_lf_delta, lf_region_spec)
                        latents_working = latents_working + latents_lf_delta
                        lf_delta_norm = float(torch.linalg.vector_norm(latents_lf_delta.to(dtype=torch.float32)).item())

                        lf_evidence = {
                            "status": "ok",
                            "coeffs_norm": float(torch.linalg.vector_norm(lf_coeffs).item()),
                            "encoded_coeffs_norm": float(torch.linalg.vector_norm(lf_coeffs_encoded).item()),
                            "lf_delta_norm": lf_delta_norm,
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

            if hf_enabled and isinstance(plan, dict):
                try:
                    basis_hf = plan.get("hf_basis")
                    hf_region_spec = plan.get("hf_region_index_spec")
                    if basis_hf is None:
                        hf_evidence = {
                            "status": "absent",
                            "absent_reason": "hf_basis_missing_in_plan"
                        }
                    else:
                        hf_coeffs = channel_hf.compute_hf_basis_projection_torch(latents_working, basis_hf)
                        hf_coeffs_constrained, constraint_evidence = channel_hf.apply_hf_truncation_constraint_torch(
                            hf_coeffs,
                            cfg
                        )
                        latents_hf_modified = channel_hf.reconstruct_from_hf_coeffs_torch(
                            hf_coeffs_constrained,
                            basis_hf,
                            original_shape,
                            dtype=original_dtype
                        )
                        latents_hf_original = channel_hf.reconstruct_from_hf_coeffs_torch(
                            hf_coeffs,
                            basis_hf,
                            original_shape,
                            dtype=original_dtype
                        )
                        latents_hf_delta = latents_hf_modified - latents_hf_original
                        latents_hf_delta = self._apply_region_spec_delta_torch(latents_hf_delta, hf_region_spec)
                        latents_working = latents_working + latents_hf_delta
                        hf_delta_norm = float(torch.linalg.vector_norm(latents_hf_delta.to(dtype=torch.float32)).item())

                        hf_evidence = {
                            "status": "ok",
                            "coeffs_norm": float(torch.linalg.vector_norm(hf_coeffs).item()),
                            "constrained_coeffs_norm": float(torch.linalg.vector_norm(hf_coeffs_constrained).item()),
                            "hf_delta_norm": hf_delta_norm,
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

            if (
                isinstance(lf_evidence, dict)
                and isinstance(hf_evidence, dict)
                and lf_evidence.get("status") == "absent"
                and hf_evidence.get("status") == "absent"
                and step_evidence.get("combined_status") == "ok"
            ):
                step_evidence["combined_status"] = "absent"

            latents_before_norm = float(
                torch.linalg.vector_norm(latents_before_modifications.to(dtype=torch.float32)).item()
            )
            latents_after_norm = float(torch.linalg.vector_norm(latents_working.to(dtype=torch.float32)).item())
            modification_delta_norm = float(
                torch.linalg.vector_norm((latents_working - latents_before_modifications).to(dtype=torch.float32)).item()
            )
            step_evidence["latents_before_norm"] = latents_before_norm
            step_evidence["latents_after_norm"] = latents_after_norm
            step_evidence["modification_delta_norm"] = modification_delta_norm
            step_evidence["lf_delta_norm"] = lf_delta_norm
            step_evidence["hf_delta_norm"] = hf_delta_norm
        except Exception as e:
            step_evidence["combined_status"] = "failed"
            step_evidence["overall_error"] = str(e)

        latents_modified = latents_working.to(device=original_device, dtype=original_dtype)
        return latents_modified, step_evidence

    def _resolve_channel_enabled(self, cfg: Dict[str, Any], channel: str) -> bool:
        """
        功能：统一解析 LF/HF 开关。

        Resolve channel enable flag from both root and watermark config.

        Args:
            cfg: Configuration mapping.
            channel: Channel tag ("lf" or "hf").

        Returns:
            Channel enabled boolean.
        """
        if channel not in {"lf", "hf"}:
            raise ValueError("channel must be lf or hf")
        direct_key = f"{channel}_enabled"
        direct_value = cfg.get(direct_key)
        if isinstance(direct_value, bool):
            return direct_value
        watermark_cfg = cfg.get("watermark") if isinstance(cfg.get("watermark"), dict) else {}
        channel_cfg = watermark_cfg.get(channel) if isinstance(watermark_cfg.get(channel), dict) else {}
        nested_value = channel_cfg.get("enabled")
        if isinstance(nested_value, bool):
            return nested_value
        return True

    def _resolve_require_basis_region_spec(self, cfg: Dict[str, Any]) -> bool:
        """
        功能：解析是否强制 basis/region_spec 校验。

        Resolve strict requirement flag for basis and region specs.

        Args:
            cfg: Configuration mapping.

        Returns:
            True when strict validation is required.
        """
        strict_flag = cfg.get("require_basis_region_spec")
        if isinstance(strict_flag, bool):
            return strict_flag
        paper_cfg = cfg.get("paper_faithfulness") if isinstance(cfg.get("paper_faithfulness"), dict) else {}
        paper_enabled = paper_cfg.get("enabled")
        return bool(paper_enabled)

    def _validate_required_plan_fields(
        self,
        plan: Optional[Dict[str, Any]],
        *,
        lf_enabled: bool,
        hf_enabled: bool,
    ) -> None:
        """
        功能：强制校验注入计划字段完整性。

        Validate basis and region spec presence for strict latent injection path.

        Args:
            plan: Plan mapping.
            lf_enabled: LF enabled flag.
            hf_enabled: HF enabled flag.

        Returns:
            None.

        Raises:
            ValueError: If required fields are missing.
        """
        if not isinstance(plan, dict):
            raise ValueError("plan must be dict when strict latent injection is enabled")
        if lf_enabled:
            if not isinstance(plan.get("lf_basis"), dict):
                raise ValueError("lf_basis is required in strict latent injection mode")
            if not isinstance(plan.get("lf_region_index_spec"), dict):
                raise ValueError("lf_region_index_spec is required in strict latent injection mode")
        if hf_enabled:
            if not isinstance(plan.get("hf_basis"), dict):
                raise ValueError("hf_basis is required in strict latent injection mode")
            if not isinstance(plan.get("hf_region_index_spec"), dict):
                raise ValueError("hf_region_index_spec is required in strict latent injection mode")

    def _apply_region_spec_delta_np(self, delta: np.ndarray, region_spec: Any) -> np.ndarray:
        """
        功能：在 numpy 路径按区域规格裁剪增量作用域。

        Apply region-index masking to numpy delta.

        Args:
            delta: Delta tensor array.
            region_spec: Region index specification mapping.

        Returns:
            Region-masked delta array.
        """
        if not isinstance(delta, np.ndarray):
            raise TypeError("delta must be np.ndarray")
        if not isinstance(region_spec, dict):
            return delta
        selected = region_spec.get("selected_indices")
        if not isinstance(selected, list):
            return delta
        flat = delta.reshape(-1)
        selected_set = {int(v) for v in selected if isinstance(v, int) and 0 <= int(v) < flat.shape[0]}
        if not selected_set:
            return np.zeros_like(delta)
        mask = np.zeros(flat.shape[0], dtype=np.float32)
        for index in sorted(selected_set):
            mask[index] = 1.0
        masked = flat * mask
        return masked.reshape(delta.shape)

    def _apply_region_spec_delta_torch(self, delta: Any, region_spec: Any) -> Any:
        """
        功能：在 torch 路径按区域规格裁剪增量作用域。

        Apply region-index masking to torch delta tensor.

        Args:
            delta: Torch delta tensor.
            region_spec: Region index specification mapping.

        Returns:
            Region-masked delta tensor.
        """
        import torch

        if not torch.is_tensor(delta):
            raise TypeError("delta must be torch.Tensor")
        if not isinstance(region_spec, dict):
            return delta
        selected = region_spec.get("selected_indices")
        if not isinstance(selected, list):
            return delta
        flat = delta.reshape(-1)
        selected_indices: List[int] = [
            int(v) for v in selected if isinstance(v, int) and 0 <= int(v) < int(flat.shape[0])
        ]
        if not selected_indices:
            return torch.zeros_like(delta)
        mask = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
        mask[torch.tensor(selected_indices, device=flat.device, dtype=torch.long)] = 1.0
        return (flat * mask).reshape(delta.shape)
