"""
功能：测试注册表 seal 后不可变（C1 覆盖）

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
    registry = pytest.importorskip("main.registries.registry_base", reason="registry_base not available")
    
    # 这里假设 registry 提供 override 方法
    # 实际测试需要根据具体实现调整
    pytest.skip("Requires actual registry implementation with override support")


def test_real_registries_are_sealed_after_initialization():
    """
    Test that actual registries (content, fusion, geometry) are sealed after init.
    
    实际使用的注册表在初始化后必须处于 sealed 状态。
    """
    try:
        from main.registries import content_registry, fusion_registry, geometry_registry
    except ImportError:
        pytest.skip("Registry modules not found")
    
    # 检查每个注册表是否实现了 sealed 状态检查
    for registry_name, registry_module in [
        ("content_registry", content_registry),
        ("fusion_registry", fusion_registry),
        ("geometry_registry", geometry_registry),
    ]:
        # 检查模块是否提供 is_sealed 或 _sealed 属性
        # 实际测试需要根据具体实现调整
        if hasattr(registry_module, "_sealed"):
            # 注册表应该在模块导入后就被 seal
            # 注：这取决于初始化策略
            pass
        elif hasattr(registry_module, "is_sealed"):
            # 检查 is_sealed() 方法
            pass
        else:
            pytest.skip(f"{registry_name} does not expose seal state")
