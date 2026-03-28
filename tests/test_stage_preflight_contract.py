"""
文件目的：验证 stage-specific preflight 的合同化门禁语义。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.workflow_acceptance_common as workflow_acceptance_common


MINIMAL_CONFIG_TEMPLATE = """
policy_path: content_np_geo_rescue
inference_prompt_file: prompts/paper_small.txt
attestation:
  enabled: {attestation_enabled}
  k_master_env_var: CEG_WM_K_MASTER
  k_prompt_env_var: CEG_WM_K_PROMPT
  k_seed_env_var: CEG_WM_K_SEED
stage_01_source_pool:
  enabled: true
  use_inference_prompt_file: true
stage_01_pooled_threshold_build:
  enabled: true
  target_pair_count: 16
""".strip()


def _write_config(path_obj: Path, *, attestation_enabled: bool = True) -> Path:
    """
    功能：写出最小 preflight 测试配置。 

    Write the minimal runtime config used by the preflight contract tests.

    Args:
        path_obj: Destination config path.
        attestation_enabled: Whether attestation is enabled in the test config.

    Returns:
        Written config path.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(
        MINIMAL_CONFIG_TEMPLATE.format(attestation_enabled=str(attestation_enabled).lower()),
        encoding="utf-8",
    )
    return path_obj


def test_detect_stage_01_preflight_fails_when_attestation_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 01 在 attestation env 缺失时必须阻断。 

    Verify stage 01 preflight fails when the attestation env vars are absent,
    even if the GPU tool is available and the config gates are otherwise valid.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = _write_config(tmp_path / "stage_01.yaml")
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    for env_name in ["CEG_WM_K_MASTER", "CEG_WM_K_PROMPT", "CEG_WM_K_SEED"]:
        monkeypatch.delenv(env_name, raising=False)

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is False
    assert preflight["gpu_tool_available"] is True
    assert preflight["missing_attestation_env_vars"] == [
        "CEG_WM_K_MASTER",
        "CEG_WM_K_PROMPT",
        "CEG_WM_K_SEED",
    ]
    assert "missing_attestation_env_vars" in preflight["failed_checks"]


def test_detect_stage_02_preflight_does_not_require_attestation_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 02 不得因为 attestation env 缺失而失败。 

    Verify stage 02 preflight remains successful when the source package and
    source contract exist, even though the attestation env vars are absent.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = _write_config(tmp_path / "stage_02.yaml")
    source_package_path = tmp_path / "source.zip"
    source_contract_path = tmp_path / "parallel_attestation_statistics_input_contract.json"
    source_package_path.write_text("zip placeholder", encoding="utf-8")
    source_contract_path.write_text("{}", encoding="utf-8")
    for env_name in ["CEG_WM_K_MASTER", "CEG_WM_K_PROMPT", "CEG_WM_K_SEED"]:
        monkeypatch.delenv(env_name, raising=False)

    preflight = workflow_acceptance_common.detect_stage_02_preflight(
        config_path,
        source_package_path,
        source_contract_path,
    )

    assert preflight["ok"] is True
    assert preflight["source_package_exists"] is True
    assert preflight["source_contract_exists"] is True
    assert preflight["missing_attestation_env_vars"] == [
        "CEG_WM_K_MASTER",
        "CEG_WM_K_PROMPT",
        "CEG_WM_K_SEED",
    ]
    assert "missing_attestation_env_vars" not in preflight["failed_checks"]


def test_detect_stage_04_preflight_does_not_require_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 04 只检查 package 输入，不得要求 GPU。 

    Verify stage 04 preflight succeeds without nvidia-smi when the required
    stage-package inputs and manifest snapshots are already prepared.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = _write_config(tmp_path / "stage_04.yaml", attestation_enabled=False)
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: None)

    stage_inputs = {
        "stage_01": {
            "status": "prepared",
            "input_path": str(tmp_path / "stage_01.zip"),
            "stage_manifest": {"stage_name": "01_Paper_Full_Cuda", "stage_run_id": "stage01"},
            "package_manifest": {"package_sha256": "sha"},
        },
        "stage_02": {
            "status": "not_provided",
            "input_path": "<absent>",
        },
        "stage_03": {
            "status": "not_provided",
            "input_path": "<absent>",
        },
    }

    preflight = workflow_acceptance_common.detect_stage_04_preflight(
        config_path,
        stage_inputs,
        require_stage_02=False,
        require_stage_03=False,
    )

    assert preflight["ok"] is True
    assert preflight["gpu_tool_available"] is False
    assert "gpu_tool_unavailable" not in preflight["failed_checks"]
