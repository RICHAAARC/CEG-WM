"""
插件能力声明与兼容性门禁

功能说明：
- 定义插件能力声明结构。
- 提供组合兼容性检查，将不兼容组合提升为写盘前可审计拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from main.core.errors import GateEnforcementError


@dataclass
class ImplCapabilities:
    """
    功能：插件能力声明结构。

    Implementation capabilities declaration.

    Args:
        supports_batching: Whether implementation supports batch processing.
        requires_cuda: Whether implementation requires CUDA.
        supports_deterministic: Whether implementation supports deterministic mode.
        max_resolution: Maximum supported resolution (e.g., "1024x1024" or None).
        supported_models: List of supported model identifiers.
    """
    supports_batching: bool = False
    requires_cuda: bool = False
    supports_deterministic: bool = True
    max_resolution: Optional[str] = None
    supported_models: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, Any]:
        """
        功能：转换为字典形式。

        Convert capabilities to dict form.

        Args:
            None.

        Returns:
            Capabilities mapping.
        """
        return asdict(self)


def assert_impl_set_compatible(impl_caps: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    功能：断言 impl_set 与 cfg 兼容。

    Assert that implementation set is compatible with configuration.
    Fail-fast on incompatible combinations.

    Args:
        impl_caps: Aggregated implementation capabilities mapping.
        cfg: Configuration mapping.

    Returns:
        None.

    Raises:
        GateEnforcementError: If capabilities and config are incompatible.
        TypeError: If inputs are invalid.
    """
    if not isinstance(impl_caps, dict):
        # impl_caps 类型不符合预期，必须 fail-fast。
        raise TypeError("impl_caps must be dict")
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    # 检查 CUDA 要求。
    requires_cuda = impl_caps.get("requires_cuda", False)
    cuda_available = cfg.get("cuda_available", True)  # 假设默认可用。

    if requires_cuda and not cuda_available:
        # CUDA 不可用但实现要求 CUDA，必须 fail-fast。
        raise GateEnforcementError(
            "capability_incompatibility: impl requires CUDA but cuda_available=False"
        )

    # 检查分辨率限制。
    max_resolution = impl_caps.get("max_resolution")
    requested_resolution = cfg.get("resolution")
    impl_ids = impl_caps.get("impl_ids")

    if requested_resolution is not None and max_resolution is not None:
        if not isinstance(requested_resolution, (int, str)):
            raise GateEnforcementError(
                "capability_incompatibility: invalid requested resolution type: "
                f"value={requested_resolution}, field_path=cfg.resolution"
            )
        if not isinstance(max_resolution, str) or not max_resolution:
            raise GateEnforcementError(
                "capability_incompatibility: invalid max_resolution declaration: "
                f"value={max_resolution}, field_path=impl_set_capabilities.max_resolution, "
                f"impl_ids={impl_ids}"
            )
        try:
            max_w, max_h = _parse_resolution(max_resolution)
            req_w, req_h = _parse_resolution(str(requested_resolution))
        except Exception as exc:
            raise GateEnforcementError(
                "capability_incompatibility: max_resolution parse failed: "
                f"value={max_resolution}, field_path=impl_set_capabilities.max_resolution, "
                f"impl_ids={impl_ids}, error={type(exc).__name__}: {exc}"
            ) from exc
        if req_w > max_w or req_h > max_h:
            raise GateEnforcementError(
                "capability_incompatibility: requested resolution exceeds max_resolution: "
                f"requested={requested_resolution}, max_resolution={max_resolution}, "
                f"impl_ids={impl_ids}"
            )

    # 检查模型支持。
    supported_models = impl_caps.get("supported_models")
    requested_model = cfg.get("model_id")

    if requested_model is not None:
        if supported_models is None:
            raise GateEnforcementError(
                "capability_incompatibility: supported_models not declared under requested model: "
                f"model_id={requested_model}, field_path=impl_set_capabilities.supported_models, "
                f"impl_ids={impl_ids}"
            )
        if not isinstance(supported_models, list):
            raise GateEnforcementError(
                "capability_incompatibility: supported_models must be list: "
                f"actual_type={type(supported_models).__name__}, "
                f"field_path=impl_set_capabilities.supported_models, impl_ids={impl_ids}"
            )
        if not supported_models:
            raise GateEnforcementError(
                "capability_incompatibility: supported_models empty under requested model: "
                f"model_id={requested_model}, field_path=impl_set_capabilities.supported_models, "
                f"impl_ids={impl_ids}"
            )
        if not isinstance(requested_model, str) or not requested_model:
            raise GateEnforcementError(
                "capability_incompatibility: invalid requested model_id: "
                f"value={requested_model}, field_path=cfg.model_id"
            )
        if requested_model not in supported_models:
            raise GateEnforcementError(
                "capability_incompatibility: model not in supported_models: "
                f"model_id={requested_model}, supported_models={supported_models}, "
                f"impl_ids={impl_ids}"
            )


def _parse_resolution(res_str: str) -> tuple[int, int]:
    """
    功能：解析分辨率字符串。

    Parse resolution string (e.g., "1024x1024" or "1024").

    Args:
        res_str: Resolution string.

    Returns:
        Tuple of (width, height).

    Raises:
        ValueError: If format is invalid.
    """
    if not isinstance(res_str, str):
        raise ValueError("res_str must be str")

    res_str = res_str.strip()
    if "x" in res_str.lower():
        parts = res_str.lower().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    # 单个数字，假设正方形。
    dim = int(res_str)
    return dim, dim
