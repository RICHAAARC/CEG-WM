"""
功能：pytest fixture 配置，为测试提供公共资源

Module type: General module

Provides common fixtures for testing: temporary directories,
minimal configuration paths, network mocking, GPU mocking.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock


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
