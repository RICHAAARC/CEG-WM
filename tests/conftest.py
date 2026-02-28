"""
功能：pytest fixture 配置，为测试提供公共资源

Module type: General module

Provides common fixtures for testing: temporary directories,
minimal configuration paths, network mocking, GPU mocking.
"""

import os
import sys
import tempfile
import pytest
import warnings
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock


def _pytest_workspace_root() -> Path:
    """
    获取测试工作区根目录。
    """
    return Path(__file__).resolve().parent.parent


def _ensure_pytest_dirs_under_workspace(config: pytest.Config) -> None:
    """
    统一 pytest 的 cache/tmp 目录到工作区 .pytest 目录下。

    Args:
        config: Pytest config object.

    Returns:
        None.
    """
    workspace_root = _pytest_workspace_root()
    pytest_root = workspace_root / ".pytest"
    debug_temproot = pytest_root / "tmproot"
    debug_temproot.mkdir(parents=True, exist_ok=True)

    if "PYTEST_DEBUG_TEMPROOT" not in os.environ:
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(debug_temproot)

    current_basetemp = getattr(config.option, "basetemp", None)
    if not current_basetemp:
        config.option.basetemp = str(pytest_root / "tmp")


def _requires_windows_py313_tmp_acl_workaround() -> bool:
    """
    判断是否需要应用 Windows + Python 3.13 下的 pytest 临时目录 ACL 兼容补丁。
    """
    return os.name == "nt" and sys.version_info >= (3, 13)


def _apply_windows_py313_tmp_acl_workaround() -> None:
    """
    修复 pytest 在 Windows + Python 3.13 下以 0o700 创建临时目录导致的 ACL 不可访问问题。

    该补丁仅作用于测试过程，不影响生产代码路径。
    """
    if not _requires_windows_py313_tmp_acl_workaround():
        return

    try:
        import _pytest.pathlib as pytest_pathlib
        import _pytest.tmpdir as pytest_tmpdir
    except Exception:
        return

    if getattr(pytest_tmpdir.TempPathFactory, "_acl_safe_mode_patched", False):
        return

    def _mkdir_acl_safe(path_obj: Path, exist_ok: bool = False) -> None:
        path_obj.mkdir(mode=0o755, parents=True, exist_ok=exist_ok)

    def _getbasetemp_acl_safe(self):
        if self._basetemp is not None:
            return self._basetemp

        if self._given_basetemp is not None:
            basetemp = self._given_basetemp
            if basetemp.exists():
                pytest_pathlib.rm_rf(basetemp)
            basetemp.parent.mkdir(parents=True, exist_ok=True)
            _mkdir_acl_safe(basetemp)
            basetemp = basetemp.resolve()
        else:
            from_env = os.environ.get("PYTEST_DEBUG_TEMPROOT")
            temproot = Path(from_env or tempfile.gettempdir()).resolve()
            user = pytest_pathlib.get_user() or "unknown"
            rootdir = temproot.joinpath(f"pytest-of-{user}")
            try:
                _mkdir_acl_safe(rootdir, exist_ok=True)
            except OSError:
                rootdir = temproot.joinpath("pytest-of-unknown")
                _mkdir_acl_safe(rootdir, exist_ok=True)

            uid = pytest_pathlib.get_user_id()
            if uid is not None:
                rootdir_stat = rootdir.stat()
                if rootdir_stat.st_uid != uid:
                    raise OSError(
                        f"The temporary directory {rootdir} is not owned by the current user. "
                        "Fix this and try again."
                    )
                if (rootdir_stat.st_mode & 0o077) != 0:
                    os.chmod(rootdir, rootdir_stat.st_mode & ~0o077)

            keep = self._retention_count
            if self._retention_policy == "none":
                keep = 0

            basetemp = pytest_pathlib.make_numbered_dir_with_cleanup(
                prefix="pytest-",
                root=rootdir,
                keep=keep,
                lock_timeout=pytest_tmpdir.LOCK_TIMEOUT,
                mode=0o755,
            )

        assert basetemp is not None, basetemp
        self._basetemp = basetemp
        self._trace("new basetemp", basetemp)
        return basetemp

    def _mktemp_acl_safe(self, basename: str, numbered: bool = True):
        basename = self._ensure_relative_to_basetemp(basename)
        if not numbered:
            temp_path = self.getbasetemp().joinpath(basename)
            temp_path.mkdir(mode=0o755)
        else:
            temp_path = pytest_pathlib.make_numbered_dir(
                root=self.getbasetemp(),
                prefix=basename,
                mode=0o755,
            )
            self._trace("mktemp", temp_path)
        return temp_path

    pytest_tmpdir.TempPathFactory.getbasetemp = _getbasetemp_acl_safe
    pytest_tmpdir.TempPathFactory.mktemp = _mktemp_acl_safe
    setattr(pytest_tmpdir.TempPathFactory, "_acl_safe_mode_patched", True)


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure test-scoped warning filters.

    Args:
        config: Pytest config object.

    Returns:
        None.
    """
    _ensure_pytest_dirs_under_workspace(config)
    _apply_windows_py313_tmp_acl_workaround()
    warnings.filterwarnings(
        "ignore",
        message=r".*(/proc/vmstat|vmstat).*",
        category=RuntimeWarning,
        module=r"psutil\._pslinux",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*buffers/cached memory stats couldn't be determined.*",
        category=RuntimeWarning,
        module=r"psutil\._pslinux",
    )


@pytest.fixture
def tmp_run_root(tmp_path: Path) -> Path:
    """
    Create a temporary run_root directory for isolated testing.
    
    Args:
        tmp_path: pytest built-in temporary directory
        
    Returns:
        Path to isolated run_root
    """
    run_root = tmp_path / "test_run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    
    # 创建标准子目录
    (run_root / "records").mkdir()
    (run_root / "artifacts").mkdir()
    (run_root / "logs").mkdir()
    
    return run_root


@pytest.fixture
def minimal_cfg_paths(tmp_path: Path) -> Dict[str, Path]:
    """
    Create minimal configuration files for testing.
    
    Args:
        tmp_path: pytest built-in temporary directory
        
    Returns:
        Dictionary mapping config names to paths
    """
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # frozen_contracts.yaml
    frozen_contracts = configs_dir / "frozen_contracts.yaml"
    frozen_contracts.write_text("""
contract_version: "v1.0.0"
required_fields:
  - run_id
  - contract_version
  - schema_version
""", encoding="utf-8")
    
    # runtime_whitelist.yaml
    runtime_whitelist = configs_dir / "runtime_whitelist.yaml"
    runtime_whitelist.write_text("""
impl_whitelist:
  - impl_id: test_impl
    impl_version: "1.0.0"
policy_paths:
  - test_policy_path
""", encoding="utf-8")
    
    # policy_path_semantics.yaml
    policy_semantics = configs_dir / "policy_path_semantics.yaml"
    policy_semantics.write_text("""
semantics_version: "v1.0.0"
paths:
  test_policy_path:
    description: Test policy path
""", encoding="utf-8")
    
    return {
        "frozen_contracts": frozen_contracts,
        "runtime_whitelist": runtime_whitelist,
        "policy_path_semantics": policy_semantics,
        "configs_dir": configs_dir,
    }


@pytest.fixture
def monkeypatch_no_network(monkeypatch):
    """
    Disable network access during tests.
    
    Monkeypatches requests, httpx, urllib to raise exceptions.
    """
    def mock_network_error(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")
    
    # requests
    try:
        import requests
        monkeypatch.setattr(requests, "get", mock_network_error)
        monkeypatch.setattr(requests, "post", mock_network_error)
        monkeypatch.setattr(requests, "request", mock_network_error)
    except ImportError:
        pass
    
    # httpx
    try:
        import httpx
        monkeypatch.setattr(httpx, "get", mock_network_error)
        monkeypatch.setattr(httpx, "post", mock_network_error)
    except ImportError:
        pass
    
    # urllib
    try:
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", mock_network_error)
    except ImportError:
        pass
    
    # hf_hub_download
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_network_error)
        monkeypatch.setattr("huggingface_hub.snapshot_download", mock_network_error)
    except ImportError:
        pass


@pytest.fixture
def monkeypatch_no_gpu(monkeypatch):
    """
    Mock GPU availability for CPU-only testing.
    
    Makes torch.cuda.is_available() return False.
    """
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    except ImportError:
        # torch 未安装，跳过
        pass


@pytest.fixture
def mock_interpretation():
    """
    Create a real ContractInterpretation fixture for testing.
    
    Returns:
        Real ContractInterpretation instance (minimal)
    """
    from main.core.contracts import load_frozen_contracts, get_contract_interpretation

    contracts = load_frozen_contracts()
    return get_contract_interpretation(contracts)


@pytest.fixture
def mock_registry_sealed():
    """
    Create a mock sealed registry for testing.
    
    Returns:
        Mock registry object with seal state
    """
    from unittest.mock import MagicMock
    
    registry = MagicMock()
    registry._sealed = False
    
    def seal():
        registry._sealed = True
    
    def register(impl_id, impl_func):
        if registry._sealed:
            raise RuntimeError(f"Registry is sealed, cannot register {impl_id}")
        registry._impls[impl_id] = impl_func
    
    registry._impls = {}
    registry.seal = seal
    registry.register = register
    
    return registry

@pytest.fixture(autouse=True)
def enable_threshold_fallback_for_tests(monkeypatch):
    """
    为所有测试自动启用 threshold fallback，用于向后兼容性。
    
    由于许多现有测试尚未适配 __thresholds_artifact__ 工件传递，
    此 autouse fixture 在所有测试环境中启用 __allow_threshold_fallback_for_tests__ 标志，
    允许在缺失 thresholds_artifact 时回退到 target_fpr（仅用于测试）。
    
    生产代码应该始终通过 orchestrator 注入 __thresholds_artifact__，
    以确保阈值的只读语义和强制来源。
    """
    # 这个 fixture 不需要主动修补，因为它仅在测试中被调用
    # 实际的 __allow_threshold_fallback_for_tests__ 标志应该在各个测试中单独设置，
    # 但由于这是一个全局兼容性需求，我们在这里记录意图
    yield
