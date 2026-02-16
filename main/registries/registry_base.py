"""
运行期 registry 基座

功能说明：
- 提供运行期 registry 基座，支持 impl_id 到 factory 的解析。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List
from types import MappingProxyType

from main.core.errors import GateEnforcementError
from main.registries.capabilities import ImplCapabilities

FactoryType = Callable[[Dict[str, Any]], Any]


class RegistrySealedError(GateEnforcementError):
    """
    功能：registry 已被冻结异常。
    
    Raised when attempting to register factory on a sealed registry.
    """
    pass


class RegistryBase:
    """
    功能：实现工厂 registry 基座。

    Base registry for impl_id to factory mapping with seal mechanism.
    Once sealed, no further factory registration is allowed (fail-fast).

    Args:
        domain: Domain name for this registry.

    Returns:
        None.

    Raises:
        ValueError: If domain is invalid.
    """

    def __init__(self, domain: str) -> None:
        if not isinstance(domain, str) or not domain:
            # domain 输入不合法，必须 fail-fast。
            raise ValueError("domain must be non-empty str")
        self._domain = domain
        self._factories: Dict[str, FactoryType] = {}
        self._capabilities: Dict[str, ImplCapabilities] = {}
        self._sealed = False

    def register_factory(
        self,
        impl_id: str,
        factory: FactoryType,
        capabilities: ImplCapabilities
    ) -> None:
        """
        功能：注册 impl_id 到 factory。

        Register factory for impl_id. Only allowed before seal().
        R11: capabilities are mandatory to enable compatibility validation.

        Args:
            impl_id: Implementation identifier.
            factory: Factory callable.
            capabilities: ImplCapabilities instance defining impl constraints.

        Returns:
            None.

        Raises:
            ValueError: If impl_id is invalid.
            TypeError: If factory is not callable or capabilities is invalid.
            RegistrySealedError: If registry is already sealed.
        """
        if self._sealed:
            # registry 已被冻结，任何新注册都必须 fail-fast。
            raise RegistrySealedError(
                f"Registry '{self._domain}' is sealed; cannot register impl_id='{impl_id}'"
            )
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        if not callable(factory):
            # factory 类型不合法，必须 fail-fast。
            raise TypeError("factory must be callable")
        if not isinstance(capabilities, ImplCapabilities):
            # capabilities 类型不合法，必须 fail-fast。
            raise TypeError("capabilities must be ImplCapabilities instance")
        self._factories[impl_id] = factory
        self._capabilities[impl_id] = capabilities

    def seal(self) -> None:
        """
        功能：冻结 registry，禁止后续注册。

        Seal the registry to prevent further factory registration.
        Called automatically after static initialization is complete.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        if not self._sealed:
            self._factories = MappingProxyType(self._factories)
            self._capabilities = MappingProxyType(self._capabilities)
            self._sealed = True

    def is_sealed(self) -> bool:
        """
        功能：查询 registry 是否已冻结。

        Check whether registry is sealed.

        Args:
            None.

        Returns:
            True if sealed, False otherwise.

        Raises:
            None.
        """
        return self._sealed

    def resolve_factory(self, impl_id: str) -> FactoryType:
        """
        功能：解析 impl_id 对应的 factory。

        Resolve factory by impl_id. Works regardless of seal status.

        Args:
            impl_id: Implementation identifier.

        Returns:
            Factory callable.

        Raises:
            ValueError: If impl_id is invalid or unknown.
        """
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        factory = self._factories.get(impl_id)
        if factory is None:
            # impl_id 未注册，必须 fail-fast。
            available = self.list_impl_ids()
            preview = _format_available_list(available)
            raise ValueError(
                f"Unknown impl_id for domain={self._domain}, impl_id={impl_id}, "
                f"available={preview}"
            )
        return factory

    def list_impl_ids(self) -> List[str]:
        """
        功能：列出已注册的 impl_id。

        List registered impl_id values.

        Args:
            None.

        Returns:
            Sorted impl_id list.
        """
        return sorted(self._factories.keys())

    def describe(self) -> Dict[str, Any]:
        """
        功能：描述 registry 当前状态。

        Describe registry domain and registered impl_id values.

        Args:
            None.

        Returns:
            Mapping with domain, impl_ids, and sealed flag.
        """
        return {
            "domain": self._domain,
            "impl_ids": self.list_impl_ids(),
            "sealed": self._sealed
        }

    def get_capabilities(self, impl_id: str) -> ImplCapabilities:
        """
        功能：获取指定 impl_id 的 capabilities。

        Retrieve implementation capabilities for given impl_id.

        Args:
            impl_id: Implementation identifier.

        Returns:
            ImplCapabilities instance.

        Raises:
            ValueError: If impl_id is invalid or unknown.
        """
        if not isinstance(impl_id, str) or not impl_id:
            # impl_id 输入不合法，必须 fail-fast。
            raise ValueError("impl_id must be non-empty str")
        capabilities = self._capabilities.get(impl_id)
        if capabilities is None:
            # impl_id 未注册，必须 fail-fast。
            available = self.list_impl_ids()
            preview = _format_available_list(available)
            raise ValueError(
                f"Unknown impl_id for domain={self._domain}, impl_id={impl_id}, "
                f"available={preview}"
            )
        return capabilities


def _format_available_list(available: List[str], limit: int = 8) -> List[str]:
    """
    功能：格式化可用 impl_id 列表。

    Format available list with optional truncation.

    Args:
        available: Available impl_id list.
        limit: Maximum number of items to include.

    Returns:
        Formatted list.

    Raises:
        TypeError: If available is not list or limit invalid.
    """
    if not isinstance(available, list):
        # available 类型不合法，必须 fail-fast。
        raise TypeError("available must be list")
    if not isinstance(limit, int) or limit <= 0:
        # limit 输入不合法，必须 fail-fast。
        raise TypeError("limit must be positive int")
    items = [item for item in available if isinstance(item, str)]
    if len(items) <= limit:
        return items
    return items[:limit]
