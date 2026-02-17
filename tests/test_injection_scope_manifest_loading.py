"""
File purpose: Injection scope manifest loader tests.
Module type: Core innovation module

功能说明：
- 验证 injection_scope_manifest.yaml 的加载与 digest 产出。
- 验证缺失必填字段时必须 fail-fast。
"""

import pytest
from pathlib import Path


def _write_manifest(path: Path, content: str) -> None:
    """
    功能：写入测试用 manifest 文件。

    Write test manifest content to file.

    Args:
        path: Target file path.
        content: YAML content.

    Returns:
        None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(path, Path):
        # path 类型不符合预期，必须 fail-fast。
        raise TypeError("path must be Path")
    if not isinstance(content, str):
        # content 类型不符合预期，必须 fail-fast。
        raise TypeError("content must be str")
    path.write_text(content, encoding="utf-8")


def test_injection_scope_manifest_loads_with_digests(tmp_path: Path) -> None:
    """
    功能：加载 manifest 并产出 digest 字段。

    Load injection_scope_manifest and verify digest fields.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        None.
    """
    from main.core.injection_scope import load_injection_scope_manifest

    manifest_path = tmp_path / "injection_scope_manifest.yaml"
    _write_manifest(
        manifest_path,
        """injection_scope_manifest_version: \"v1\"
allowed_impl_ids:
  - \"impl_test\"
digest_scope:
  cfg_digest_include_paths:
    - \"cfg.policy_path\"
  plan_digest_include_paths:
    - \"plan.impl_id\"
versioning_policy:
  bump_manifest_version_on_digest_scope_change: true
  bump_manifest_version_on_impl_scope_change: true
frozen_surface_protected_files:
  - \"configs/frozen_contracts.yaml\"
operational_notes: \"test\"
"""
    )

    manifest = load_injection_scope_manifest(
        str(manifest_path),
        allow_non_authoritative=True
    )

    assert manifest.injection_scope_manifest_version == "v1"
    assert isinstance(manifest.injection_scope_manifest_digest, str)
    assert isinstance(manifest.injection_scope_manifest_file_sha256, str)
    assert isinstance(manifest.injection_scope_manifest_canon_sha256, str)
    assert isinstance(manifest.injection_scope_manifest_bound_digest, str)


def test_injection_scope_manifest_missing_version_fails(tmp_path: Path) -> None:
    """
    功能：缺失版本字段时必须失败。

    Fail fast when injection_scope_manifest_version is missing.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        None.
    """
    from main.core.injection_scope import load_injection_scope_manifest
    from main.core.errors import MissingRequiredFieldError

    manifest_path = tmp_path / "injection_scope_manifest.yaml"
    _write_manifest(
        manifest_path,
        """allowed_impl_ids:
  - \"impl_test\"
digest_scope:
  cfg_digest_include_paths:
    - \"cfg.policy_path\"
  plan_digest_include_paths:
    - \"plan.impl_id\"
versioning_policy:
  bump_manifest_version_on_digest_scope_change: true
  bump_manifest_version_on_impl_scope_change: true
frozen_surface_protected_files:
  - \"configs/frozen_contracts.yaml\"
"""
    )

    with pytest.raises(MissingRequiredFieldError):
        load_injection_scope_manifest(
            str(manifest_path),
            allow_non_authoritative=True
        )


def test_injection_scope_manifest_real_config_loads() -> None:
    """
    功能：真实配置文件必须可加载。

    Load real configs/injection_scope_manifest.yaml and ensure parsing succeeds.

    Args:
      None.

    Returns:
      None.
    """
    from main.core.injection_scope import load_injection_scope_manifest

    manifest = load_injection_scope_manifest()
    assert isinstance(manifest.injection_scope_manifest_digest, str)
    assert isinstance(manifest.injection_scope_manifest_file_sha256, str)
    assert isinstance(manifest.injection_scope_manifest_canon_sha256, str)
    assert isinstance(manifest.injection_scope_manifest_bound_digest, str)
