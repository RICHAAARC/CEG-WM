"""
File purpose: Runtime resolver capabilities digest v2 回归测试。
Module type: General module
"""

from __future__ import annotations

from main.core import config_loader
from main.registries import runtime_resolver


def _load_cfg(path: str) -> dict:
    cfg, _ = config_loader.load_yaml_with_provenance(path)
    return cfg


def test_runtime_resolver_sets_capabilities_v2_digest_for_default_profile() -> None:
    cfg = _load_cfg("configs/default.yaml")
    _, _, digest_v1 = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)
    digest_extended = cfg.get("impl_set_capabilities_extended_digest")

    assert isinstance(digest_v1, str)
    assert len(digest_v1) == 64
    assert isinstance(digest_extended, str)
    assert len(digest_extended) == 64


def test_runtime_resolver_sets_capabilities_v2_digest_for_paper_profile() -> None:
    cfg = _load_cfg("configs/paper_full_cuda.yaml")
    _, _, digest_v1 = runtime_resolver.build_runtime_impl_set_from_cfg(cfg)
    digest_extended = cfg.get("impl_set_capabilities_extended_digest")

    assert isinstance(digest_v1, str)
    assert len(digest_v1) == 64
    assert isinstance(digest_extended, str)
    assert len(digest_extended) == 64
    assert digest_v1 != digest_extended
