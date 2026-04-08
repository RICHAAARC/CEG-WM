"""
文件目的：验证 stage-specific preflight 的合同化门禁语义。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.workflow_acceptance_common as workflow_acceptance_common


def _write_config(
    path_obj: Path,
    *,
    attestation_enabled: bool = True,
    model_snapshot_path: Path | None = None,
    binding_status: str = "bound",
    binding_reason: str | None = None,
    binding_snapshot_path: str | None = None,
    experiment_matrix_config_path: str | None = None,
) -> Path:
    """
    功能：写出最小 preflight 测试配置。 

    Write the minimal runtime config used by the preflight contract tests.

    Args:
        path_obj: Destination config path.
        attestation_enabled: Whether attestation is enabled in the test config.
        model_snapshot_path: Optional runtime-bound model snapshot path.
        binding_status: Binding status written into model_source_binding.
        binding_reason: Optional binding reason override.
        binding_snapshot_path: Optional snapshot path stored inside
            model_source_binding.

    Returns:
        Written config path.
    """
    if model_snapshot_path is not None and not isinstance(model_snapshot_path, Path):
        raise TypeError("model_snapshot_path must be Path or None")
    if not isinstance(binding_status, str) or not binding_status:
        raise TypeError("binding_status must be non-empty str")
    if binding_reason is not None and (not isinstance(binding_reason, str) or not binding_reason):
        raise TypeError("binding_reason must be non-empty str or None")
    if binding_snapshot_path is not None and (
        not isinstance(binding_snapshot_path, str) or not binding_snapshot_path
    ):
        raise TypeError("binding_snapshot_path must be non-empty str or None")
    if experiment_matrix_config_path is not None and (
        not isinstance(experiment_matrix_config_path, str) or not experiment_matrix_config_path
    ):
        raise TypeError("experiment_matrix_config_path must be non-empty str or None")

    cfg_obj = {
        "policy_path": "content_np_geo_rescue",
        "inference_prompt_file": "prompts/paper_small.txt",
        "attestation": {
            "enabled": attestation_enabled,
            "k_master_env_var": "CEG_WM_K_MASTER",
            "k_prompt_env_var": "CEG_WM_K_PROMPT",
            "k_seed_env_var": "CEG_WM_K_SEED",
        },
        "stage_01_source_pool": {
            "enabled": True,
            "use_inference_prompt_file": True,
        },
        "stage_01_pooled_threshold_build": {
            "enabled": True,
            "target_pair_count": 16,
        },
    }
    if experiment_matrix_config_path is not None:
        cfg_obj["experiment_matrix"] = {
            "config_path": experiment_matrix_config_path,
        }
    if model_snapshot_path is not None:
        snapshot_path_text = model_snapshot_path.as_posix()
        resolved_binding_reason = binding_reason or (
            "model_snapshot_env_var_bound_to_runtime_config"
            if binding_status == "bound"
            else "model_snapshot_env_var_path_missing_or_not_directory"
        )
        cfg_obj.update(
            {
                "model_id": "stabilityai/stable-diffusion-3.5-medium",
                "model_source": "hf",
                "hf_revision": "main",
                "model_snapshot_path": snapshot_path_text,
                "model_source_binding": {
                    "binding_source": "notebook_snapshot_download",
                    "binding_env_var": "CEG_WM_MODEL_SNAPSHOT_PATH",
                    "binding_status": binding_status,
                    "binding_reason": resolved_binding_reason,
                    "model_snapshot_path": binding_snapshot_path or snapshot_path_text,
                    "requested_model_id": "stabilityai/stable-diffusion-3.5-medium",
                    "requested_model_source": "hf",
                    "requested_hf_revision": "main",
                },
            }
        )

    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(yaml.safe_dump(cfg_obj, sort_keys=False), encoding="utf-8")
    return path_obj


def _set_attestation_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    功能：为 preflight 测试补齐最小 attestation env。 

    Set the minimum attestation environment variables required by stage-01
    preflight tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setenv("CEG_WM_K_MASTER", "a" * 64)
    monkeypatch.setenv("CEG_WM_K_PROMPT", "b" * 32)
    monkeypatch.setenv("CEG_WM_K_SEED", "c" * 32)


def test_detect_stage_01_preflight_passes_with_bound_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 01 在模型绑定有效时必须通过 formal preflight。 

    Verify stage-01 preflight passes when the notebook-bound model snapshot is
    present, directory-backed, and internally consistent.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(tmp_path / "stage_01_bound.yaml", model_snapshot_path=snapshot_dir)
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    _set_attestation_env_vars(monkeypatch)

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is True
    assert preflight["model_source_binding_present"] is True
    assert preflight["model_source_binding_status"] == "bound"
    assert preflight["model_snapshot_path"] == snapshot_dir.as_posix()
    assert preflight["model_snapshot_path_exists"] is True
    assert preflight["model_snapshot_path_is_directory"] is True
    assert preflight["model_source_binding_path_matches_snapshot_path"] is True
    assert preflight["failed_checks"] == []


def test_detect_stage_01_preflight_fails_when_whitelist_semantics_versions_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 01 在正式 whitelist 与 semantics 版本不一致时必须提前失败。

    Verify stage-01 preflight fails early when the authoritative runtime
    whitelist version does not match the policy semantics version.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(tmp_path / "stage_01_version_mismatch.yaml", model_snapshot_path=snapshot_dir)
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    _set_attestation_env_vars(monkeypatch)
    monkeypatch.setattr(
        workflow_acceptance_common,
        "_collect_authoritative_policy_binding_summary",
        lambda: {
            "runtime_whitelist_version": "v2.4",
            "policy_path_semantics_version": "v2.6",
            "whitelist_semantics_versions_match": False,
        },
    )

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is False
    assert preflight["runtime_whitelist_version"] == "v2.4"
    assert preflight["policy_path_semantics_version"] == "v2.6"
    assert preflight["whitelist_semantics_versions_match"] is False
    assert "policy_semantics_whitelist_version_mismatch" in preflight["failed_checks"]


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
    snapshot_dir = tmp_path / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(tmp_path / "stage_01.yaml", model_snapshot_path=snapshot_dir)
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
    assert "stage_01_model_source_binding_missing" not in preflight["failed_checks"]
    assert "stage_01_model_snapshot_path_missing_or_not_directory" not in preflight["failed_checks"]


def test_detect_stage_01_preflight_fails_when_model_source_binding_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 01 在模型来源绑定缺失时必须阻断。 

    Verify stage-01 preflight hard-fails when the runtime config snapshot does
    not carry the notebook model-source binding fields.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = _write_config(tmp_path / "stage_01_missing_binding.yaml")
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    _set_attestation_env_vars(monkeypatch)

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is False
    assert preflight["model_source_binding_present"] is False
    assert preflight["model_source_binding_status"] == "<absent>"
    assert "stage_01_model_source_binding_missing" in preflight["failed_checks"]
    assert "stage_01_model_snapshot_path_missing_or_not_directory" in preflight["failed_checks"]


def test_detect_stage_01_preflight_fails_when_model_source_binding_not_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 01 在模型来源绑定状态非 bound 时必须阻断。 

    Verify stage-01 preflight hard-fails when model_source_binding exists but
    does not declare a bound runtime snapshot.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(
        tmp_path / "stage_01_invalid_binding.yaml",
        model_snapshot_path=snapshot_dir,
        binding_status="invalid",
        binding_reason="model_snapshot_env_var_path_missing_or_not_directory",
    )
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    _set_attestation_env_vars(monkeypatch)

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is False
    assert preflight["model_source_binding_status"] == "invalid"
    assert preflight["model_snapshot_path_exists"] is True
    assert preflight["model_snapshot_path_is_directory"] is True
    assert "stage_01_model_source_binding_not_bound" in preflight["failed_checks"]
    assert "stage_01_model_snapshot_path_missing_or_not_directory" not in preflight["failed_checks"]


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_detect_stage_01_preflight_fails_when_model_snapshot_path_is_not_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_kind: str,
) -> None:
    """
    功能：stage 01 在模型快照路径不存在或非目录时必须阻断。 

    Verify stage-01 preflight hard-fails when the bound model snapshot path is
    absent or resolves to a file instead of a directory.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.
        path_kind: Whether the configured path is absent or file-backed.

    Returns:
        None.
    """
    snapshot_path = tmp_path / f"snapshot_{path_kind}"
    if path_kind == "file":
        snapshot_path.write_text("not a directory", encoding="utf-8")
    config_path = _write_config(
        tmp_path / f"stage_01_snapshot_{path_kind}.yaml",
        model_snapshot_path=snapshot_path,
    )
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    _set_attestation_env_vars(monkeypatch)

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is False
    assert preflight["model_source_binding_status"] == "bound"
    assert preflight["model_snapshot_path_is_directory"] is False
    assert "stage_01_model_snapshot_path_missing_or_not_directory" in preflight["failed_checks"]


def test_detect_stage_01_preflight_fails_when_binding_path_mismatches_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 01 在绑定路径与运行时快照路径不一致时必须阻断。 

    Verify stage-01 preflight hard-fails when model_source_binding points to a
    different snapshot directory than the top-level runtime config field.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "model_snapshot"
    binding_snapshot_dir = tmp_path / "binding_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    binding_snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(
        tmp_path / "stage_01_binding_mismatch.yaml",
        model_snapshot_path=snapshot_dir,
        binding_snapshot_path=binding_snapshot_dir.as_posix(),
    )
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    _set_attestation_env_vars(monkeypatch)

    preflight = workflow_acceptance_common.detect_stage_01_preflight(config_path)

    assert preflight["ok"] is False
    assert preflight["model_snapshot_path_exists"] is True
    assert preflight["model_snapshot_path_is_directory"] is True
    assert preflight["model_source_binding_path_matches_snapshot_path"] is False
    assert "stage_01_model_source_binding_path_mismatch" in preflight["failed_checks"]


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


def test_detect_stage_03_preflight_fails_when_model_snapshot_binding_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 在 runtime config 缺少有效模型绑定时必须 fail-fast。

    Verify stage-03 preflight hard-fails when the runtime config lacks a valid
    model snapshot binding.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    config_path = _write_config(tmp_path / "stage_03_missing_binding.yaml", attestation_enabled=False)
    source_package_path = tmp_path / "stage_01_source.zip"
    source_thresholds_artifact_path = tmp_path / "thresholds_artifact.json"
    source_package_path.write_text("zip placeholder", encoding="utf-8")
    source_thresholds_artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")

    preflight = workflow_acceptance_common.detect_stage_03_preflight(
        config_path,
        source_package_path,
        source_thresholds_artifact_path,
        require_model_binding=True,
        require_authoritative_config_path=True,
    )

    assert preflight["ok"] is False
    assert preflight["model_source_binding_required"] is True
    assert preflight["model_source_binding_present"] is False
    assert preflight["model_source_binding_status"] == "<absent>"
    assert "stage_03_model_source_binding_missing" in preflight["failed_checks"]
    assert "stage_03_model_snapshot_path_missing_or_not_directory" in preflight["failed_checks"]
    assert "stage_03_experiment_matrix_config_path_missing" in preflight["failed_checks"]


def test_detect_stage_03_preflight_fails_when_attestation_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 在 attestation 开启且环境变量缺失时必须 fail-fast。

    Verify stage-03 preflight hard-fails when attestation is enabled but the
    required secret environment variables are absent.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "stage_03_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(
        tmp_path / "stage_03_missing_attestation_env.yaml",
        attestation_enabled=True,
        model_snapshot_path=snapshot_dir,
        experiment_matrix_config_path=(tmp_path / "stage_03_missing_attestation_env.yaml").as_posix(),
    )
    source_package_path = tmp_path / "stage_01_source.zip"
    source_thresholds_artifact_path = tmp_path / "thresholds_artifact.json"
    source_package_path.write_text("zip placeholder", encoding="utf-8")
    source_thresholds_artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")
    monkeypatch.delenv("CEG_WM_K_MASTER", raising=False)
    monkeypatch.delenv("CEG_WM_K_PROMPT", raising=False)
    monkeypatch.delenv("CEG_WM_K_SEED", raising=False)

    preflight = workflow_acceptance_common.detect_stage_03_preflight(
        config_path,
        source_package_path,
        source_thresholds_artifact_path,
        require_model_binding=True,
        require_authoritative_config_path=True,
    )

    assert preflight["ok"] is False
    assert preflight["attestation_env_required"] is True
    assert preflight["attestation_env_var_bindings_complete"] is True
    assert preflight["missing_attestation_env_vars"] == [
        "CEG_WM_K_MASTER",
        "CEG_WM_K_PROMPT",
        "CEG_WM_K_SEED",
    ]
    assert "missing_attestation_env_vars" in preflight["failed_checks"]
    assert "attestation_env_var_bindings_incomplete" not in preflight["failed_checks"]


def test_detect_stage_03_preflight_passes_with_bound_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 在 runtime config 含有效模型绑定时必须通过 preflight。

    Verify stage-03 preflight passes when the runtime config carries a valid
    bound model snapshot directory.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "stage_03_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(
        tmp_path / "stage_03_bound_binding.yaml",
        attestation_enabled=False,
        model_snapshot_path=snapshot_dir,
        experiment_matrix_config_path=(tmp_path / "stage_03_bound_binding.yaml").as_posix(),
    )
    source_package_path = tmp_path / "stage_01_source.zip"
    source_thresholds_artifact_path = tmp_path / "thresholds_artifact.json"
    source_package_path.write_text("zip placeholder", encoding="utf-8")
    source_thresholds_artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")

    preflight = workflow_acceptance_common.detect_stage_03_preflight(
        config_path,
        source_package_path,
        source_thresholds_artifact_path,
        require_model_binding=True,
        require_authoritative_config_path=True,
    )

    assert preflight["ok"] is True
    assert preflight["model_source_binding_required"] is True
    assert preflight["model_source_binding_present"] is True
    assert preflight["model_source_binding_status"] == "bound"
    assert preflight["model_snapshot_path"] == snapshot_dir.as_posix()
    assert preflight["model_snapshot_path_exists"] is True
    assert preflight["model_snapshot_path_is_directory"] is True
    assert preflight["model_source_binding_path_matches_snapshot_path"] is True
    assert preflight["experiment_matrix_config_path"] == config_path.resolve().as_posix()
    assert preflight["experiment_matrix_config_path_matches_runtime_snapshot"] is True
    assert preflight["failed_checks"] == []


def test_detect_stage_03_preflight_fails_when_experiment_matrix_config_path_mismatches_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：stage 03 在 experiment_matrix.config_path 未指向当前 runtime snapshot 时必须 fail-fast。

    Verify stage-03 preflight hard-fails when experiment_matrix.config_path
    points to a non-authoritative config path such as configs/default.yaml.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    snapshot_dir = tmp_path / "stage_03_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(
        tmp_path / "stage_03_mismatched_matrix_config.yaml",
        attestation_enabled=False,
        model_snapshot_path=snapshot_dir,
        experiment_matrix_config_path="configs/default.yaml",
    )
    source_package_path = tmp_path / "stage_01_source.zip"
    source_thresholds_artifact_path = tmp_path / "thresholds_artifact.json"
    source_package_path.write_text("zip placeholder", encoding="utf-8")
    source_thresholds_artifact_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workflow_acceptance_common.shutil, "which", lambda _command: "/usr/bin/nvidia-smi")

    preflight = workflow_acceptance_common.detect_stage_03_preflight(
        config_path,
        source_package_path,
        source_thresholds_artifact_path,
        require_model_binding=True,
        require_authoritative_config_path=True,
    )

    assert preflight["ok"] is False
    assert preflight["model_source_binding_status"] == "bound"
    assert preflight["experiment_matrix_config_path"].endswith("configs/default.yaml")
    assert preflight["experiment_matrix_config_path_matches_runtime_snapshot"] is False
    assert "stage_03_experiment_matrix_config_path_mismatch" in preflight["failed_checks"]


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
