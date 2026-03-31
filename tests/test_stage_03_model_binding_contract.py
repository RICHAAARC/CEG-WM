"""
文件目的：验证 stage 03 模型快照 binding 的 notebook 优先级与 source-stage fallback 合同。
Module type: General module
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_03_SCRIPT_PATH = REPO_ROOT / "scripts" / "03_Experiment_Matrix_Full.py"


def _load_stage_03_module() -> ModuleType:
    """
    功能：按文件路径加载 stage 03 脚本模块。

    Load the stage-03 wrapper module from its filesystem path.

    Args:
        None.

    Returns:
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location("test_stage_03_model_binding_contract_module", STAGE_03_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {STAGE_03_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_stage_03_cfg() -> Dict[str, Any]:
    """
    功能：构造 stage 03 最小运行配置。

    Build the minimal stage-03 runtime configuration used by the tests.

    Args:
        None.

    Returns:
        Minimal configuration mapping.
    """
    return {
        "policy_path": "content_np_geo_rescue",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_source": "hf",
        "hf_revision": "main",
    }


def _write_source_runtime_config(
    path_obj: Path,
    snapshot_path: Path,
    *,
    binding_source: str,
) -> Path:
    """
    功能：写出 source stage runtime config snapshot。

    Write the source-stage runtime config snapshot used for fallback tests.

    Args:
        path_obj: Destination config path.
        snapshot_path: Source snapshot directory.
        binding_source: Binding source string to persist.

    Returns:
        Written config path.
    """
    payload = {
        "policy_path": "content_np_geo_rescue",
        "model_id": "stabilityai/stable-diffusion-3.5-medium",
        "model_source": "hf",
        "hf_revision": "main",
        "model_snapshot_path": snapshot_path.as_posix(),
        "model_source_binding": {
            "binding_source": binding_source,
            "binding_env_var": "CEG_WM_MODEL_SNAPSHOT_PATH",
            "binding_status": "bound",
            "binding_reason": "model_snapshot_env_var_bound_to_runtime_config",
            "model_snapshot_path": snapshot_path.as_posix(),
            "requested_model_id": "stabilityai/stable-diffusion-3.5-medium",
            "requested_model_source": "hf",
            "requested_hf_revision": "main",
        },
    }
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path_obj


def test_stage_03_runtime_config_prefers_notebook_model_snapshot_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 runtime config 必须优先采用 notebook 传入的模型绑定。

    Verify stage-03 runtime config resolution prefers the notebook-provided
    model snapshot binding over the source-stage fallback.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_03_module = _load_stage_03_module()
    notebook_snapshot_dir = tmp_path / "notebook_snapshot"
    source_snapshot_dir = tmp_path / "source_snapshot"
    notebook_snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_runtime_config_path = _write_source_runtime_config(
        tmp_path / "source_runtime_config_snapshot.yaml",
        source_snapshot_dir,
        binding_source="source_stage_runtime_config_snapshot",
    )
    monkeypatch.setenv("CEG_WM_MODEL_SNAPSHOT_PATH", notebook_snapshot_dir.as_posix())

    resolved_cfg = stage_03_module._resolve_stage_03_runtime_config(
        _base_stage_03_cfg(),
        source_runtime_config_path,
    )
    runtime_cfg = stage_03_module._build_runtime_config(
        resolved_cfg,
        tmp_path / "run_root",
        tmp_path / "readonly_thresholds.json",
    )

    assert runtime_cfg["model_snapshot_path"] == notebook_snapshot_dir.resolve().as_posix()
    assert runtime_cfg["model_source_binding"]["binding_status"] == "bound"
    assert runtime_cfg["model_source_binding"]["binding_source"] == "notebook_snapshot_download"
    assert runtime_cfg["model_source_binding"]["binding_env_var"] == "CEG_WM_MODEL_SNAPSHOT_PATH"
    assert runtime_cfg["model_source_binding"]["model_snapshot_path"] == notebook_snapshot_dir.resolve().as_posix()


def test_stage_03_runtime_config_falls_back_to_source_runtime_snapshot_when_notebook_binding_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：当 notebook binding 缺失时，stage 03 runtime config 必须从 source runtime config 回退继承。

    Verify stage-03 runtime config falls back to the source-stage runtime
    config snapshot when the notebook binding is absent.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    stage_03_module = _load_stage_03_module()
    source_snapshot_dir = tmp_path / "source_snapshot"
    source_snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_runtime_config_path = _write_source_runtime_config(
        tmp_path / "source_runtime_config_snapshot.yaml",
        source_snapshot_dir,
        binding_source="source_stage_runtime_config_snapshot",
    )
    monkeypatch.delenv("CEG_WM_MODEL_SNAPSHOT_PATH", raising=False)

    resolved_cfg = stage_03_module._resolve_stage_03_runtime_config(
        _base_stage_03_cfg(),
        source_runtime_config_path,
    )
    runtime_cfg = stage_03_module._build_runtime_config(
        resolved_cfg,
        tmp_path / "run_root",
        tmp_path / "readonly_thresholds.json",
    )

    assert runtime_cfg["model_snapshot_path"] == source_snapshot_dir.as_posix()
    assert runtime_cfg["model_source_binding"]["binding_status"] == "bound"
    assert runtime_cfg["model_source_binding"]["binding_source"] == "source_stage_runtime_config_snapshot"
    assert runtime_cfg["model_source_binding"]["model_snapshot_path"] == source_snapshot_dir.as_posix()