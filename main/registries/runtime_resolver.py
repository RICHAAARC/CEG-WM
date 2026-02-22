"""
运行期 impl 解析与装配

功能说明：
- 解析 impl_identity 中的 impl_id 到对应的 factory。
- 使用解析到的 factory 构造运行期 impl 对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List

from main.core import digests
from main.core.errors import MissingRequiredFieldError, GateEnforcementError
from main.registries import content_registry, geometry_registry, fusion_registry
from main.registries import impl_identity as impl_identity_module
from main.registries.capabilities import ImplCapabilities, assert_impl_set_compatible


@dataclass(frozen=True)
class ImplIdentity:
    """
    功能：实现身份字段集合。

    Runtime impl identity mapping.

    Args:
        content_extractor_id: Content extractor impl_id.
        geometry_extractor_id: Geometry extractor impl_id.
        fusion_rule_id: Fusion rule impl_id.
        subspace_planner_id: Subspace planner impl_id.
        sync_module_id: Sync module impl_id.

    Returns:
        None.

    Raises:
        ValueError: If any field is invalid.
    """

    content_extractor_id: str
    geometry_extractor_id: str
    fusion_rule_id: str
    subspace_planner_id: str
    sync_module_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.content_extractor_id, str) or not self.content_extractor_id:
            # content_extractor_id 输入不合法，必须 fail-fast。
            raise ValueError("content_extractor_id must be non-empty str")
        if not isinstance(self.geometry_extractor_id, str) or not self.geometry_extractor_id:
            # geometry_extractor_id 输入不合法，必须 fail-fast。
            raise ValueError("geometry_extractor_id must be non-empty str")
        if not isinstance(self.fusion_rule_id, str) or not self.fusion_rule_id:
            # fusion_rule_id 输入不合法，必须 fail-fast。
            raise ValueError("fusion_rule_id must be non-empty str")
        if not isinstance(self.subspace_planner_id, str) or not self.subspace_planner_id:
            # subspace_planner_id 输入不合法，必须 fail-fast。
            raise ValueError("subspace_planner_id must be non-empty str")
        if not isinstance(self.sync_module_id, str) or not self.sync_module_id:
            # sync_module_id 输入不合法，必须 fail-fast。
            raise ValueError("sync_module_id must be non-empty str")

    def as_dict(self) -> Dict[str, str]:
        """
        功能：导出 impl_identity 映射。

        Export impl identity as dict.

        Args:
            None.

        Returns:
            Mapping of impl identity fields.
        """
        return {
            "content_extractor_id": self.content_extractor_id,
            "geometry_extractor_id": self.geometry_extractor_id,
            "fusion_rule_id": self.fusion_rule_id,
            "subspace_planner_id": self.subspace_planner_id,
            "sync_module_id": self.sync_module_id
        }


@dataclass(frozen=True)
class ImplFactorySpec:
    """
    功能：factory 规格。

    Factory specification for a domain.

    Args:
        domain: Domain name.
        impl_id: Implementation identifier.
        factory: Factory callable.

    Returns:
        None.
    """

    domain: str
    impl_id: str
    factory: Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class ResolvedImplFactories:
    """
    功能：解析后的 factory 集合。

    Collection of resolved factories without construction.

    Args:
        content_extractor: Content extractor factory spec.
        geometry_extractor: Geometry extractor factory spec.
        fusion_rule: Fusion rule factory spec.
        subspace_planner: Subspace planner factory spec.
        sync_module: Sync module factory spec.

    Returns:
        None.
    """

    content_extractor: ImplFactorySpec
    geometry_extractor: ImplFactorySpec
    fusion_rule: ImplFactorySpec
    subspace_planner: ImplFactorySpec
    sync_module: ImplFactorySpec


@dataclass(frozen=True)
class BuiltImplSet:
    """
    功能：已构造实现对象集合。

    Collection of built runtime implementations.

    Args:
        content_extractor: Built content extractor.
        geometry_extractor: Built geometry extractor.
        fusion_rule: Built fusion rule.
        subspace_planner: Built subspace planner.
        sync_module: Built sync module.

    Returns:
        None.
    """

    content_extractor: Any
    geometry_extractor: Any
    fusion_rule: Any
    subspace_planner: Any
    sync_module: Any


def parse_impl_identity_from_cfg(cfg: Dict[str, Any]) -> ImplIdentity:
    """
    功能：从 cfg 解析 impl_identity。

    Parse impl identity from config mapping.

    Args:
        cfg: Config mapping.

    Returns:
        ImplIdentity object.

    Raises:
        MissingRequiredFieldError: If required fields are missing or invalid.
        TypeError: If cfg or impl section is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    impl_cfg = cfg.get("impl")
    if not isinstance(impl_cfg, dict):
        # impl 缺失或类型错误，必须 fail-fast。
        raise MissingRequiredFieldError("Missing required field: impl")

    return ImplIdentity(
        content_extractor_id=_require_str_field(impl_cfg, "content_extractor_id", "impl.content_extractor_id"),
        geometry_extractor_id=_require_str_field(impl_cfg, "geometry_extractor_id", "impl.geometry_extractor_id"),
        fusion_rule_id=_require_str_field(impl_cfg, "fusion_rule_id", "impl.fusion_rule_id"),
        subspace_planner_id=_require_str_field(impl_cfg, "subspace_planner_id", "impl.subspace_planner_id"),
        sync_module_id=_require_str_field(impl_cfg, "sync_module_id", "impl.sync_module_id")
    )


def resolve_impl_factories(
    identity: ImplIdentity
) -> ResolvedImplFactories:
    """
    功能：解析各域 factory。

    Resolve factory specs for each impl domain.
    Uses statically sealed registries; no runtime override is permitted.

    Args:
        identity: Impl identity mapping.

    Returns:
        ResolvedImplFactories instance.

    Raises:
        ValueError: If identity or factory resolution fails.
        TypeError: If inputs are invalid.
    """
    if not isinstance(identity, ImplIdentity):
        # identity 类型不合法，必须 fail-fast。
        raise TypeError("identity must be ImplIdentity")

    # 直接使用模块级的静态 registry，不允许 caller 注入替代品。
    content_factory = _resolve_factory(
        content_registry.resolve_content_extractor,
        "content_extractor",
        identity.content_extractor_id
    )
    geometry_factory = _resolve_factory(
        geometry_registry.resolve_geometry_extractor,
        "geometry_extractor",
        identity.geometry_extractor_id
    )
    fusion_factory = _resolve_factory(
        fusion_registry.resolve_fusion_rule,
        "fusion_rule",
        identity.fusion_rule_id
    )
    subspace_factory = _resolve_factory(
        content_registry.resolve_subspace_planner,
        "subspace_planner",
        identity.subspace_planner_id
    )
    sync_factory = _resolve_factory(
        geometry_registry.resolve_sync_module,
        "sync_module",
        identity.sync_module_id
    )

    return ResolvedImplFactories(
        content_extractor=ImplFactorySpec("content_extractor", identity.content_extractor_id, content_factory),
        geometry_extractor=ImplFactorySpec("geometry_extractor", identity.geometry_extractor_id, geometry_factory),
        fusion_rule=ImplFactorySpec("fusion_rule", identity.fusion_rule_id, fusion_factory),
        subspace_planner=ImplFactorySpec("subspace_planner", identity.subspace_planner_id, subspace_factory),
        sync_module=ImplFactorySpec("sync_module", identity.sync_module_id, sync_factory)
    )


def build_impl_set(resolved: ResolvedImplFactories, cfg: Dict[str, Any]) -> BuiltImplSet:
    """
    功能：构造实现对象集合。

    Build runtime implementations from resolved factories.

    Args:
        resolved: Resolved factory specs.
        cfg: Config mapping.

    Returns:
        BuiltImplSet instance.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If factory construction fails.
    """
    if not isinstance(resolved, ResolvedImplFactories):
        # resolved 类型不合法，必须 fail-fast。
        raise TypeError("resolved must be ResolvedImplFactories")
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    content_extractor = _build_impl(resolved.content_extractor, cfg)
    geometry_extractor = _build_impl(resolved.geometry_extractor, cfg)
    fusion_rule = _build_impl(resolved.fusion_rule, cfg)
    subspace_planner = _build_impl(resolved.subspace_planner, cfg)
    sync_module = _build_impl(resolved.sync_module, cfg)

    return BuiltImplSet(
        content_extractor=content_extractor,
        geometry_extractor=geometry_extractor,
        fusion_rule=fusion_rule,
        subspace_planner=subspace_planner,
        sync_module=sync_module
    )


def build_runtime_impl_set_from_cfg(cfg: Dict[str, Any]) -> tuple[ImplIdentity, BuiltImplSet, str]:
    """
    功能：从 cfg 构建运行期 impl_set。

    Build runtime impl identity and implementation set from config.

    Args:
        cfg: Config mapping.

    Returns:
        Tuple of (ImplIdentity, BuiltImplSet, impl_set_capabilities_digest).

    Raises:
        GateEnforcementError: If impl set capabilities are incompatible.
        Exception: Propagates parse/resolve/build errors.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")

    identity = parse_impl_identity_from_cfg(cfg)
    resolved = resolve_impl_factories(identity)
    impl_set = build_impl_set(resolved, cfg)

    # 聚合所有 impl 的 capabilities，并执行兼容性校验。
    impl_caps_list: List[ImplCapabilities] = []
    impl_caps_list.append(
        content_registry._CONTENT_REGISTRY.get_capabilities(identity.content_extractor_id)
    )
    impl_caps_list.append(
        geometry_registry._GEOMETRY_REGISTRY.get_capabilities(identity.geometry_extractor_id)
    )
    impl_caps_list.append(
        fusion_registry._FUSION_REGISTRY.get_capabilities(identity.fusion_rule_id)
    )
    impl_caps_list.append(
        content_registry._SUBSPACE_REGISTRY.get_capabilities(identity.subspace_planner_id)
    )
    impl_caps_list.append(
        geometry_registry._SYNC_REGISTRY.get_capabilities(identity.sync_module_id)
    )

    # 调用 assert_impl_set_compatible 执行门禁检查。
    impl_caps_gate = _aggregate_impl_capabilities_for_gate(
        impl_caps_list,
        cfg,
        [
            identity.content_extractor_id,
            identity.geometry_extractor_id,
            identity.fusion_rule_id,
            identity.subspace_planner_id,
            identity.sync_module_id
        ]
    )
    assert_impl_set_compatible(impl_caps_gate, cfg)

    # 计算 impl_set_capabilities_digest（仅保留可审计字段：bool/enum/str-list）。
    impl_set_capabilities_digest = compute_impl_set_capabilities_digest(impl_caps_list)

    return identity, impl_set, impl_set_capabilities_digest


def _aggregate_impl_capabilities_for_gate(
    impl_caps_list: List[ImplCapabilities],
    cfg: Dict[str, Any],
    impl_ids: List[str]
) -> Dict[str, Any]:
    """
    Aggregate per-impl capabilities into a single gate payload.
    Rejects undeclared/contradictory capability states under requested constraints.
    """
    if not isinstance(impl_caps_list, list) or not impl_caps_list:
        raise TypeError("impl_caps_list must be non-empty list")
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(impl_ids, list) or len(impl_ids) != len(impl_caps_list):
        raise TypeError("impl_ids must be list with same length as impl_caps_list")
    if not all(isinstance(item, ImplCapabilities) for item in impl_caps_list):
        raise TypeError("impl_caps_list items must be ImplCapabilities")
    if not all(isinstance(item, str) and item for item in impl_ids):
        raise TypeError("impl_ids items must be non-empty str")

    aggregated: Dict[str, Any] = {}
    aggregated["impl_ids"] = list(impl_ids)
    aggregated["requires_cuda"] = any(cap.requires_cuda is True for cap in impl_caps_list)
    aggregated["supports_batching"] = all(cap.supports_batching is True for cap in impl_caps_list)
    aggregated["supports_deterministic"] = all(
        cap.supports_deterministic is True for cap in impl_caps_list
    )

    requested_model = cfg.get("model_id")
    if requested_model is not None:
        if not isinstance(requested_model, str) or not requested_model:
            raise GateEnforcementError(
                "capability constraint invalid: "
                "capability_name=supported_models, field_path=cfg.model_id, "
                f"requested_value={requested_model}"
            )
        undeclared_impl_ids = [
            impl_ids[idx]
            for idx, cap in enumerate(impl_caps_list)
            if cap.supported_models is None
        ]
        if undeclared_impl_ids:
            raise GateEnforcementError(
                "capability undeclared under requested constraint: "
                "capability_name=supported_models, "
                "field_path=impl_set_capabilities.supported_models, "
                f"impl_ids={undeclared_impl_ids}, requested_model={requested_model}"
            )

        model_sets: List[set[str]] = []
        for idx, cap in enumerate(impl_caps_list):
            if not isinstance(cap.supported_models, list):
                raise GateEnforcementError(
                    "capability declaration invalid: "
                    "capability_name=supported_models, "
                    "field_path=impl_set_capabilities.supported_models, "
                    f"impl_ids={[impl_ids[idx]]}, actual_type={type(cap.supported_models).__name__}"
                )
            if not all(isinstance(item, str) and item for item in cap.supported_models):
                raise GateEnforcementError(
                    "capability declaration invalid: "
                    "capability_name=supported_models, "
                    "field_path=impl_set_capabilities.supported_models, "
                    f"impl_ids={[impl_ids[idx]]}, detail=non_str_or_empty_item"
                )
            model_sets.append(set(cap.supported_models))

        supported_models = sorted(set.intersection(*model_sets))
        if not supported_models:
            raise GateEnforcementError(
                "capability incompatibility: "
                "capability_name=supported_models, "
                "field_path=impl_set_capabilities.supported_models, "
                f"impl_ids={impl_ids}, requested_model={requested_model}, intersection=[]"
            )
        aggregated["supported_models"] = supported_models

    requested_resolution = cfg.get("resolution")
    declared_resolutions: List[tuple[int, int]] = []
    declared_resolution_impl_ids: List[str] = []
    undeclared_resolution_impl_ids: List[str] = []
    for idx, cap in enumerate(impl_caps_list):
        if cap.max_resolution is None:
            undeclared_resolution_impl_ids.append(impl_ids[idx])
            continue
        try:
            declared_resolutions.append(_parse_resolution_for_gate(cap.max_resolution))
            declared_resolution_impl_ids.append(impl_ids[idx])
        except ValueError as exc:
            raise GateEnforcementError(
                "capability declaration invalid: "
                "capability_name=max_resolution, "
                "field_path=impl_set_capabilities.max_resolution, "
                f"impl_ids={[impl_ids[idx]]}, detail={exc}"
            ) from exc

    if requested_resolution is not None and undeclared_resolution_impl_ids:
        raise GateEnforcementError(
            "capability undeclared under requested constraint: "
            "capability_name=max_resolution, "
            "field_path=impl_set_capabilities.max_resolution, "
            f"impl_ids={undeclared_resolution_impl_ids}, requested_resolution={requested_resolution}"
        )

    if declared_resolutions:
        agg_w = min(value[0] for value in declared_resolutions)
        agg_h = min(value[1] for value in declared_resolutions)
        aggregated["max_resolution"] = f"{agg_w}x{agg_h}"
    elif requested_resolution is not None:
        raise GateEnforcementError(
            "capability undeclared under requested constraint: "
            "capability_name=max_resolution, "
            "field_path=impl_set_capabilities.max_resolution, "
            f"impl_ids={impl_ids}, requested_resolution={requested_resolution}"
        )

    return aggregated


def _parse_resolution_for_gate(value: Any) -> tuple[int, int]:
    """
    Parse resolution value to (width, height) for gate aggregation.
    Supports "WxH" and "W".
    """
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip().lower()
    else:
        raise ValueError(f"resolution must be str|int, got {type(value).__name__}")

    if not text:
        raise ValueError("resolution must be non-empty")

    if "x" in text:
        parts = text.split("x")
        if len(parts) != 2:
            raise ValueError(f"invalid resolution format: {value}")
        try:
            width = int(parts[0].strip())
            height = int(parts[1].strip())
        except Exception as exc:
            raise ValueError(f"invalid resolution number: {value}") from exc
    else:
        try:
            width = int(text)
            height = int(text)
        except Exception as exc:
            raise ValueError(f"invalid resolution number: {value}") from exc

    if width <= 0 or height <= 0:
        raise ValueError(f"resolution must be positive: {value}")
    return width, height


def compute_impl_identity_digest(identity: ImplIdentity) -> str:
    """
    功能：计算 impl_identity 摘要。

    Compute impl identity digest from impl_id mapping.

    Args:
        identity: Impl identity object.

    Returns:
        Canonical digest string.

    Raises:
        TypeError: If identity is invalid.
    """
    if not isinstance(identity, ImplIdentity):
        # identity 类型不合法，必须 fail-fast。
        raise TypeError("identity must be ImplIdentity")
    return digests.canonical_sha256(identity.as_dict())


def compute_impl_set_capabilities_digest(impl_caps_list: List[ImplCapabilities]) -> str:
    """
    功能：计算 impl_set capabilities 摘要。

    Compute impl_set_capabilities_digest from aggregated capabilities.
    Only includes auditable fields (bool/enum/str-list) for stability.

    Args:
        impl_caps_list: List of ImplCapabilities instances.

    Returns:
        Canonical digest string.

    Raises:
        TypeError: If input is invalid.
    """
    if not isinstance(impl_caps_list, list):
        # impl_caps_list 类型不合法，必须 fail-fast。
        raise TypeError("impl_caps_list must be list")
    
    # 构造 canonical dict。
    domains = [
        "content_extractor",
        "geometry_extractor",
        "fusion_rule",
        "subspace_planner",
        "sync_module"
    ]
    
    if len(impl_caps_list) != len(domains):
        raise ValueError(f"impl_caps_list length mismatch: expected {len(domains)}, got {len(impl_caps_list)}")
    
    canonical_caps: Dict[str, Any] = {}
    for idx, domain in enumerate(domains):
        caps = impl_caps_list[idx]
        if not isinstance(caps, ImplCapabilities):
            raise TypeError(f"impl_caps_list[{idx}] must be ImplCapabilities")
        
        # 仅保留可审计字段。
        domain_caps: Dict[str, Any] = {}
        domain_caps["supports_batching"] = caps.supports_batching
        domain_caps["requires_cuda"] = caps.requires_cuda
        domain_caps["supports_deterministic"] = caps.supports_deterministic
        
        if caps.max_resolution is not None and isinstance(caps.max_resolution, str):
            domain_caps["max_resolution"] = caps.max_resolution
        
        if caps.supported_models is not None and isinstance(caps.supported_models, list):
            domain_caps["supported_models"] = sorted(caps.supported_models)
        
        canonical_caps[domain] = domain_caps
    
    return digests.canonical_sha256(canonical_caps)


def _require_str_field(mapping: Dict[str, Any], key: str, field_path: str) -> str:
    """
    功能：读取并校验必需字符串字段。

    Require a non-empty string field from mapping.

    Args:
        mapping: Mapping to read.
        key: Field key.
        field_path: Field path string for error reporting.

    Returns:
        Field value string.

    Raises:
        MissingRequiredFieldError: If field missing or invalid.
        TypeError: If mapping is not dict.
    """
    if not isinstance(mapping, dict):
        # mapping 类型不合法，必须 fail-fast。
        raise TypeError("mapping must be dict")
    if not isinstance(key, str) or not key:
        # key 输入不合法，必须 fail-fast。
        raise TypeError("key must be non-empty str")
    if not isinstance(field_path, str) or not field_path:
        # field_path 输入不合法，必须 fail-fast。
        raise TypeError("field_path must be non-empty str")

    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        # 字段缺失或类型错误，必须 fail-fast。
        raise MissingRequiredFieldError(f"Missing required field: {field_path}")
    return value


def _resolve_factory(
    resolver: Callable[[str], Callable[[Dict[str, Any]], Any]],
    domain: str,
    impl_id: str
) -> Callable[[Dict[str, Any]], Any]:
    """
    功能：解析 domain 对应 factory。

    Resolve factory with consistent error context.

    Args:
        resolver: Resolver callable.
        domain: Domain name.
        impl_id: Implementation identifier.

    Returns:
        Factory callable.

    Raises:
        ValueError: If resolution fails.
    """
    if not callable(resolver):
        # resolver 类型不合法，必须 fail-fast。
        raise TypeError("resolver must be callable")
    if not isinstance(domain, str) or not domain:
        # domain 输入不合法，必须 fail-fast。
        raise TypeError("domain must be non-empty str")
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 输入不合法，必须 fail-fast。
        raise TypeError("impl_id must be non-empty str")

    try:
        return resolver(impl_id)
    except Exception as exc:
        # 解析失败必须 fail-fast，并给出 domain/impl_id。
        raise ValueError(
            f"Failed to resolve impl_id for domain={domain}, impl_id={impl_id}: {exc}"
        ) from exc


def _build_impl(spec: ImplFactorySpec, cfg: Dict[str, Any]) -> Any:
    """
    功能：构造单个实现对象。

    Build a single implementation from spec.

    Args:
        spec: Factory specification.
        cfg: Config mapping.

    Returns:
        Built implementation object.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If construction fails.
    """
    if not isinstance(spec, ImplFactorySpec):
        # spec 类型不合法，必须 fail-fast。
        raise TypeError("spec must be ImplFactorySpec")
    if not isinstance(cfg, dict):
        # cfg 类型不合法，必须 fail-fast。
        raise TypeError("cfg must be dict")
    if not callable(spec.factory):
        # factory 类型不合法，必须 fail-fast。
        raise TypeError("spec.factory must be callable")

    try:
        built = spec.factory(cfg)
    except Exception as exc:
        # 构造失败必须 fail-fast，并给出 domain/impl_id。
        raise ValueError(
            f"Failed to build impl for domain={spec.domain}, impl_id={spec.impl_id}: {exc}"
        ) from exc

    _validate_built_impl_metadata(built, spec.domain, spec.impl_id)
    return built


def _validate_built_impl_metadata(obj: Any, domain: str, impl_id: str) -> None:
    """
    功能：校验构造对象元信息。

    Validate built impl object has required metadata fields.

    Args:
        obj: Built implementation object.
        domain: Domain name.
        impl_id: Implementation identifier.

    Returns:
        None.

    Raises:
        ValueError: If metadata missing or invalid.
    """
    if not isinstance(domain, str) or not domain:
        # domain 输入不合法，必须 fail-fast。
        raise TypeError("domain must be non-empty str")
    if not isinstance(impl_id, str) or not impl_id:
        # impl_id 输入不合法，必须 fail-fast。
        raise TypeError("impl_id must be non-empty str")

    impl_version = getattr(obj, "impl_version", None)
    impl_digest = getattr(obj, "impl_digest", None)
    if not isinstance(impl_version, str) or not impl_version:
        # impl_version 缺失，必须 fail-fast。
        raise ValueError(
            f"Built impl missing impl_version for domain={domain}, impl_id={impl_id}"
        )
    if not isinstance(impl_digest, str) or not impl_digest:
        # impl_digest 缺失，必须 fail-fast。
        raise ValueError(
            f"Built impl missing impl_digest for domain={domain}, impl_id={impl_id}"
        )


def build_impl_meta(
    identity: ImplIdentity,
    built_set: BuiltImplSet
) -> impl_identity_module.ImplMeta:
    """
    功能：从已构造实现对象构造 ImplMeta。
    
    Build ImplMeta by extracting version and digest from built runtime implementations.
    
    Args:
        identity: ImplIdentity containing impl_id selections.
        built_set: BuiltImplSet containing constructed runtime impl objects.
    
    Returns:
        ImplMeta instance with all versions and digests populated.
    
    Raises:
        TypeError: If inputs are invalid.
        ValueError: If impl objects missing metadata.
    """
    if not isinstance(identity, ImplIdentity):
        raise TypeError(f"identity must be ImplIdentity, got {type(identity)}")
    if not isinstance(built_set, BuiltImplSet):
        raise TypeError(f"built_set must be BuiltImplSet, got {type(built_set)}")
    
    # 从各实现对象中提取版本与摘要。
    items = {
        "content_extractor": (built_set.content_extractor, "content_extractor"),
        "geometry_extractor": (built_set.geometry_extractor, "geometry_extractor"),
        "fusion_rule": (built_set.fusion_rule, "fusion_rule"),
        "subspace_planner": (built_set.subspace_planner, "subspace_planner"),
        "sync_module": (built_set.sync_module, "sync_module"),
    }
    
    extracted = {}
    for key, (impl_obj, domain_name) in items.items():
        impl_version = getattr(impl_obj, "impl_version", None)
        impl_digest = getattr(impl_obj, "impl_digest", None)
        
        if not isinstance(impl_version, str) or not impl_version:
            raise ValueError(
                f"Built impl {domain_name} missing or invalid impl_version"
            )
        if not isinstance(impl_digest, str) or not impl_digest:
            raise ValueError(
                f"Built impl {domain_name} missing or invalid impl_digest"
            )
        
        extracted[f"{key}_version"] = impl_version
        extracted[f"{key}_digest"] = impl_digest
    
    # 转换为 ImplIdentity 实例用于 ImplMeta 构造。
    identity_for_meta = impl_identity_module.ImplIdentity(
        content_extractor_id=identity.content_extractor_id,
        geometry_extractor_id=identity.geometry_extractor_id,
        fusion_rule_id=identity.fusion_rule_id,
        subspace_planner_id=identity.subspace_planner_id,
        sync_module_id=identity.sync_module_id
    )
    
    return impl_identity_module.ImplMeta(
        impl_identity=identity_for_meta,
        content_extractor_version=extracted["content_extractor_version"],
        content_extractor_digest=extracted["content_extractor_digest"],
        geometry_extractor_version=extracted["geometry_extractor_version"],
        geometry_extractor_digest=extracted["geometry_extractor_digest"],
        fusion_rule_version=extracted["fusion_rule_version"],
        fusion_rule_digest=extracted["fusion_rule_digest"],
        subspace_planner_version=extracted["subspace_planner_version"],
        subspace_planner_digest=extracted["subspace_planner_digest"],
        sync_module_version=extracted["sync_module_version"],
        sync_module_digest=extracted["sync_module_digest"]
    )
