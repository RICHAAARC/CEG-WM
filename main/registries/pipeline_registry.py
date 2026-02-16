"""
管道壳注册表

功能说明：
- 管道壳注册表用于受控注入管道实现。
- 提供静态注册、seal 与 resolve 入口，禁止运行期修改。
"""

from __future__ import annotations

from typing import Any, Dict

from main.core import digests

from .registry_base import FactoryType, RegistryBase
from .capabilities import ImplCapabilities


SD3_DIFFUSERS_SHELL_ID = "sd3_diffusers_shell_v1"
SD3_DIFFUSERS_REAL_ID = "sd3_diffusers_real_v1"


class PipelineShellPlaceholder:
    """
    功能：管道壳占位实现。

    Placeholder pipeline shell implementation with metadata only.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version string.
        impl_digest: Implementation digest string.

    Returns:
        None.

    Raises:
        ValueError: If any input is invalid.
    """

    def __init__(self, impl_id: str, impl_version: str, impl_digest: str) -> None:
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        if not isinstance(impl_version, str) or not impl_version:
            # impl_version 输入不合法，必须 fail-fast。
            raise ValueError("impl_version must be non-empty str")
        if not isinstance(impl_digest, str) or not impl_digest:
            # impl_digest 输入不合法，必须 fail-fast。
            raise ValueError("impl_digest must be non-empty str")
        self.impl_id = impl_id
        self.impl_version = impl_version
        self.impl_digest = impl_digest

    def as_dict(self) -> Dict[str, str]:
        """
        功能：导出占位实现元数据。

        Export placeholder metadata as dict.

        Args:
            None.

        Returns:
            Mapping with impl_id, impl_version, and impl_digest.
        """
        return {
            "pipeline_impl_id": self.impl_id,
            "pipeline_impl_version": self.impl_version,
            "pipeline_impl_digest": self.impl_digest
        }


_PIPELINE_REGISTRY = RegistryBase("pipeline_shell")


def _derive_impl_digest(impl_id: str, impl_version: str) -> str:
    """
    功能：计算 pipeline impl_digest。

    Compute pipeline impl_digest from impl_id and impl_version.

    Args:
        impl_id: Implementation identifier.
        impl_version: Implementation version string.

    Returns:
        Canonical digest string.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 输入不合法，必须 fail-fast。
        raise ValueError("impl_id must be non-empty str")
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 输入不合法，必须 fail-fast。
        raise ValueError("impl_version must be non-empty str")
    return digests.canonical_sha256({
        "impl_id": impl_id,
        "impl_version": impl_version
    })


def _build_sd3_diffusers_shell(cfg: Dict[str, Any]) -> PipelineShellPlaceholder:
    """
    功能：构造 SD3 管道壳占位实现。

    Build SD3 pipeline shell placeholder.

    Args:
        cfg: Config mapping.

    Returns:
        PipelineShellPlaceholder instance.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(SD3_DIFFUSERS_SHELL_ID, impl_version)
    return PipelineShellPlaceholder(SD3_DIFFUSERS_SHELL_ID, impl_version, impl_digest)


def _build_sd3_diffusers_real(cfg: Dict[str, Any]) -> PipelineShellPlaceholder:
    """
    功能：构造 SD3 真实 pipeline 占位实现。

    Build SD3 real pipeline placeholder.

    Args:
        cfg: Config mapping.

    Returns:
        PipelineShellPlaceholder instance.

    Raises:
        TypeError: If cfg is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    impl_version = "v1"
    impl_digest = _derive_impl_digest(SD3_DIFFUSERS_REAL_ID, impl_version)
    return PipelineShellPlaceholder(SD3_DIFFUSERS_REAL_ID, impl_version, impl_digest)


_PIPELINE_REGISTRY.register_factory(
    SD3_DIFFUSERS_SHELL_ID,
    _build_sd3_diffusers_shell,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=None
    )
)

_PIPELINE_REGISTRY.register_factory(
    SD3_DIFFUSERS_REAL_ID,
    _build_sd3_diffusers_real,
    capabilities=ImplCapabilities(
        supports_batching=False,
        requires_cuda=False,
        supports_deterministic=True,
        max_resolution=None,
        supported_models=None
    )
)

# 静态注册完成后立即冻结，禁止运行期修改。
_PIPELINE_REGISTRY.seal()


def resolve_pipeline_shell(impl_id: str) -> FactoryType:
    """
    功能：解析 pipeline_impl_id。

    Resolve pipeline shell factory.

    Args:
        impl_id: Pipeline implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If impl_id is invalid or unknown.
    """
    return _PIPELINE_REGISTRY.resolve_factory(impl_id)


def list_pipeline_impl_ids() -> list[str]:
    """
    功能：列出 pipeline_impl_id。

    List pipeline impl_id values.

    Args:
        None.

    Returns:
        List of pipeline impl_id values.
    """
    return _PIPELINE_REGISTRY.list_impl_ids()
