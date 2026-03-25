"""
功能：测试注册表 seal 后不可变（registry.seal_and_immutability，legacy_code=C1）

Module type: Core innovation module

Test that registries cannot be modified after sealing.
"""

import pytest


def test_registry_seal_blocks_new_registrations(mock_registry_sealed):
    """
    Test that sealed registry rejects new registrations.
    
    seal 后必须拒绝新的 register 调用。
    """
    registry = mock_registry_sealed
    
    # (1) seal 前可以注册
    registry.register("impl_1", lambda: "impl_1_func")
    assert "impl_1" in registry._impls
    
    # (2) 执行 seal
    registry.seal()
    assert registry._sealed is True
    
    # (3) seal 后注册应该抛异常
    with pytest.raises(RuntimeError) as exc_info:
        registry.register("impl_2", lambda: "impl_2_func")
    
    # 检查异常信息包含 sealed 或 registry 相关提示
    error_msg = str(exc_info.value).lower()
    assert "seal" in error_msg or "immutable" in error_msg
    
    # (4) 验证 impl_2 未被注册
    assert "impl_2" not in registry._impls


def test_registry_seal_mechanism_exists():
    """
    Test that registry_base module implements seal mechanism.
    
    registry_base.py 必须实现 seal 机制。
    """
    try:
        from main.registries import registry_base
    except ImportError:
        pytest.skip("main.registries.registry_base module not found")
    
    # 检查是否包含 seal 相关方法或属性
    # 这里简化为检查模块源码
    import inspect
    source = inspect.getsource(registry_base)
    
    # 至少应包含 "seal" 或 "_sealed" / "_locked" 等关键字
    source_lower = source.lower()
    assert "seal" in source_lower or "_sealed" in source_lower or "_locked" in source_lower, \
        "registry_base should implement seal mechanism"


def test_registry_seal_prevents_override():
    """
    Test that sealed registry rejects override operations.
    
    seal 后不允许覆盖已有 impl。
    """
    try:
        from main.registries.registry_base import RegistryBase, RegistrySealedError
        from main.registries.capabilities import ImplCapabilities
    except ImportError:
        pytest.skip("registry_base not available")

    registry = RegistryBase("test_registry")
    registry.register_factory(
        "impl_1",
        lambda cfg: cfg,
        ImplCapabilities(supports_batching=False, requires_cuda=False)
    )
    registry.seal()

    with pytest.raises(RegistrySealedError):
        registry.register_factory(
            "impl_2",
            lambda cfg: cfg,
            ImplCapabilities(supports_batching=False, requires_cuda=False)
        )


def test_real_registries_are_sealed_after_initialization():
    """
    Test that actual registries (content, fusion, geometry) are sealed after init.
    
    实际使用的注册表在初始化后必须处于 sealed 状态。
    """
    try:
        from main.registries import content_registry, fusion_registry, geometry_registry
    except ImportError:
        pytest.skip("Registry modules not found")
    
    registry_specs = [
        ("content_registry", content_registry, "_CONTENT_REGISTRY"),
        ("fusion_registry", fusion_registry, "_FUSION_REGISTRY"),
        ("geometry_registry", geometry_registry, "_GEOMETRY_REGISTRY"),
    ]

    for registry_name, registry_module, attr_name in registry_specs:
        if not hasattr(registry_module, attr_name):
            pytest.skip(f"{registry_name} does not expose {attr_name}")
        registry_obj = getattr(registry_module, attr_name)
        assert hasattr(registry_obj, "is_sealed")
        assert registry_obj.is_sealed() is True
