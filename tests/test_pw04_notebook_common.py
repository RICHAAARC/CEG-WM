"""
File purpose: Contract tests for shared PW04 notebook helper logic.
Module type: General module
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from paper_workflow.scripts.pw04_notebook_common import (
    build_gpu_peak_notebook_summary,
    build_pw04_command,
    build_pw04_subprocess_env,
    load_gpu_peak_summary,
    read_pw04_result_summary,
    resolve_pw04_expected_output,
    resolve_pw04_quality_runtime_summary,
)
from paper_workflow.scripts.pw_quality_metrics import (
    DEFAULT_QUALITY_BATCH_SIZE,
    QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET_ENV,
    QUALITY_PSNR_SSIM_BATCH_SIZE_ENV,
)


def _write_json(path_obj: Path, payload: Dict[str, Any]) -> None:
    """
    Write one JSON object for tests.

    Args:
        path_obj: Output path.
        payload: JSON payload.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path_obj: Path, text: str) -> None:
    """
    Write one UTF-8 text file for tests.

    Args:
        path_obj: Output path.
        text: File content.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(text, encoding="utf-8")


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool, device_count: int = 0) -> None:
    """
    Install one fake torch module for deterministic runtime-summary tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        cuda_available: Whether CUDA should appear available.
        device_count: Visible CUDA device count.

    Returns:
        None.
    """
    if cuda_available:
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: device_count,
            get_device_name=lambda index: f"Fake GPU {index}",
            get_device_properties=lambda index: SimpleNamespace(total_memory=10 * 1024 ** 3),
        )
    else:
        fake_cuda = SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))


def test_resolve_pw04_quality_runtime_summary_auto_returns_valid_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify auto mode returns a valid runtime summary structure.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=False)

    summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="auto",
        base_env={},
    )

    assert summary["requested_device"] == "auto"
    assert summary["selected_device"] == "cpu"
    assert summary["lpips_batch_size"] == DEFAULT_QUALITY_BATCH_SIZE
    assert summary["clip_batch_size"] == DEFAULT_QUALITY_BATCH_SIZE
    assert summary["lpips_batch_size_source"] == "device_default"
    assert summary["clip_batch_size_source"] == "device_default"
    assert summary["warnings"] == []
    assert summary["torch_runtime_status"] == "imported"


def test_resolve_pw04_quality_runtime_summary_invalid_device_falls_back_and_reads_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify invalid device overrides fall back and environment batch sizes are applied.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=False)

    summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="bogus",
        base_env={
            "PW_QUALITY_LPIPS_BATCH_SIZE": "3",
            "PW_QUALITY_CLIP_BATCH_SIZE": "5",
        },
    )

    assert summary["requested_device"] == "auto"
    assert summary["selected_device"] == "cpu"
    assert summary["lpips_batch_size"] == 3
    assert summary["clip_batch_size"] == 5
    assert summary["lpips_batch_size_source"] == "environment"
    assert summary["clip_batch_size_source"] == "environment"
    assert any("QUALITY_DEVICE_OVERRIDE" in warning for warning in summary["warnings"])


def test_resolve_pw04_quality_runtime_summary_cuda_defaults_use_fixed_gpu_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify CUDA defaults always use the fixed GPU batch sizes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=True, device_count=1)

    summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="auto",
        base_env={},
    )

    assert summary["selected_device"] == "cuda"
    assert summary["detected_cuda_total_memory_gib"] == 10.0
    assert summary["lpips_batch_size"] == 128
    assert summary["clip_batch_size"] == 256
    assert summary["lpips_batch_size_source"] == "device_default"
    assert summary["clip_batch_size_source"] == "device_default"
    assert "conservative" not in summary["batch_default_reason"]


def test_resolve_pw04_quality_runtime_summary_notebook_override_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify notebook batch overrides take precedence over environment values.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=True, device_count=1)

    summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="auto",
        quality_lpips_batch_size_override=9,
        quality_clip_batch_size_override="11",
        base_env={
            "PW_QUALITY_LPIPS_BATCH_SIZE": "3",
            "PW_QUALITY_CLIP_BATCH_SIZE": "5",
        },
    )

    assert summary["lpips_batch_size"] == 9
    assert summary["clip_batch_size"] == 11
    assert summary["lpips_batch_size_source"] == "notebook_override"
    assert summary["clip_batch_size_source"] == "notebook_override"


def test_resolve_pw04_quality_runtime_summary_invalid_notebook_override_falls_back_to_device_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify invalid notebook overrides fall back to device defaults with warnings.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=True, device_count=1)

    summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="auto",
        quality_lpips_batch_size_override=0,
        quality_clip_batch_size_override="invalid",
        base_env={
            "PW_QUALITY_LPIPS_BATCH_SIZE": "7",
            "PW_QUALITY_CLIP_BATCH_SIZE": "13",
        },
    )

    assert summary["lpips_batch_size"] == 128
    assert summary["clip_batch_size"] == 256
    assert summary["lpips_batch_size_source"] == "device_default"
    assert summary["clip_batch_size_source"] == "device_default"
    assert any("QUALITY_LPIPS_BATCH_SIZE" in warning for warning in summary["warnings"])
    assert any("QUALITY_CLIP_BATCH_SIZE" in warning for warning in summary["warnings"])


def test_build_pw04_subprocess_env_injects_repo_and_quality_runtime(tmp_path: Path) -> None:
    """
    Verify subprocess env includes repo import context and quality runtime variables.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    env_mapping = build_pw04_subprocess_env(
        repo_root=tmp_path,
        base_env={"PATH": "demo-path", "PYTHONPATH": "existing-path"},
        pw04_mode="quality_shard",
        quality_runtime_summary={
            "selected_device": "cuda",
            "lpips_batch_size": 7,
            "clip_batch_size": 11,
            "psnr_ssim_batch_size_override": None,
            "psnr_ssim_batch_element_budget_override": None,
        },
    )

    assert env_mapping["PW_QUALITY_TORCH_DEVICE"] == "cuda"
    assert env_mapping["PW_QUALITY_LPIPS_BATCH_SIZE"] == "7"
    assert env_mapping["PW_QUALITY_CLIP_BATCH_SIZE"] == "11"
    assert QUALITY_PSNR_SSIM_BATCH_SIZE_ENV not in env_mapping
    assert QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET_ENV not in env_mapping
    assert str(tmp_path.resolve()) in env_mapping["PYTHONPATH"]


def test_build_pw04_subprocess_env_injects_psnr_ssim_batch_size_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify notebook PSNR/SSIM batch-size override is injected into the subprocess environment.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=False)

    quality_runtime_summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="auto",
        quality_psnr_ssim_batch_size_override=16,
        quality_psnr_ssim_batch_element_budget_override=None,
        base_env={},
    )
    env_mapping = build_pw04_subprocess_env(
        repo_root=tmp_path,
        base_env={"PATH": "demo-path"},
        pw04_mode="quality_shard",
        quality_runtime_summary=quality_runtime_summary,
    )

    assert env_mapping[QUALITY_PSNR_SSIM_BATCH_SIZE_ENV] == "16"
    assert QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET_ENV not in env_mapping


def test_build_pw04_subprocess_env_injects_psnr_ssim_batch_element_budget_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify notebook PSNR/SSIM batch-element-budget override is injected into the subprocess environment.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    _install_fake_torch(monkeypatch, cuda_available=False)

    quality_runtime_summary = resolve_pw04_quality_runtime_summary(
        quality_device_override="auto",
        quality_psnr_ssim_batch_size_override=None,
        quality_psnr_ssim_batch_element_budget_override=12582912,
        base_env={},
    )
    env_mapping = build_pw04_subprocess_env(
        repo_root=tmp_path,
        base_env={"PATH": "demo-path"},
        pw04_mode="quality_shard",
        quality_runtime_summary=quality_runtime_summary,
    )

    assert QUALITY_PSNR_SSIM_BATCH_SIZE_ENV not in env_mapping
    assert env_mapping[QUALITY_PSNR_SSIM_BATCH_ELEMENT_BUDGET_ENV] == "12582912"


def test_build_pw04_subprocess_env_prepare_mode_skips_quality_runtime_binding(tmp_path: Path) -> None:
    """
    Verify prepare mode leaves quality runtime variables unbound.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    env_mapping = build_pw04_subprocess_env(
        repo_root=tmp_path,
        base_env={"PATH": "demo-path", "PYTHONPATH": "existing-path"},
        pw04_mode="prepare",
    )

    assert "PW_QUALITY_TORCH_DEVICE" not in env_mapping
    assert "PW_QUALITY_LPIPS_BATCH_SIZE" not in env_mapping
    assert "PW_QUALITY_CLIP_BATCH_SIZE" not in env_mapping
    assert str(tmp_path.resolve()) in env_mapping["PYTHONPATH"]


@pytest.mark.parametrize(
    ("pw04_mode", "quality_shard_count", "expected_has_quality_index", "expected_has_quality_count"),
    [
        ("prepare", None, False, False),
        ("prepare", 3, False, True),
        ("quality_shard", None, True, False),
        ("finalize", None, False, False),
    ],
)
def test_build_pw04_command_handles_mode_specific_flags(
    tmp_path: Path,
    pw04_mode: str,
    quality_shard_count: int | None,
    expected_has_quality_index: bool,
    expected_has_quality_count: bool,
) -> None:
    """
    Verify command construction preserves mode-specific flags.

    Args:
        tmp_path: Pytest temporary directory.
        pw04_mode: PW04 mode token.
        expected_has_quality_index: Whether the quality-shard flag is expected.

    Returns:
        None.
    """
    command = build_pw04_command(
        script_path=tmp_path / "PW04_Attack_Merge_And_Metrics.py",
        drive_project_root=tmp_path / "drive_root",
        family_id="family_demo",
        pw04_mode=pw04_mode,
        quality_shard_index=7,
        quality_shard_count=quality_shard_count,
        force_rerun=True,
        enable_tail_estimation=True,
    )

    assert str(tmp_path / "PW04_Attack_Merge_And_Metrics.py") in command
    assert "--drive-project-root" in command
    assert "--family-id" in command
    assert "--pw04-mode" in command
    assert ("--quality-shard-index" in command) is expected_has_quality_index
    assert ("7" in command) is expected_has_quality_index
    assert ("--quality-shard-count" in command) is expected_has_quality_count
    assert ("3" in command) is expected_has_quality_count
    assert "--force-rerun" in command
    assert "--enable-tail-estimation" in command


@pytest.mark.parametrize("pw04_mode", ["quality_shard", "finalize"])
def test_build_pw04_command_rejects_quality_shard_count_outside_prepare(
    tmp_path: Path,
    pw04_mode: str,
) -> None:
    """
    Verify explicit quality_shard_count is rejected outside prepare mode.

    Args:
        tmp_path: Pytest temporary directory.
        pw04_mode: Non-prepare PW04 mode token.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="quality_shard_count is only valid"):
        build_pw04_command(
            script_path=tmp_path / "PW04_Attack_Merge_And_Metrics.py",
            drive_project_root=tmp_path / "drive_root",
            family_id="family_demo",
            pw04_mode=pw04_mode,
            quality_shard_index=7,
            quality_shard_count=2,
            force_rerun=False,
            enable_tail_estimation=False,
        )


def test_resolve_pw04_expected_output_maps_modes_to_paths(tmp_path: Path) -> None:
    """
    Verify the expected-output helper maps each mode to the correct label and path.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    prepare_manifest_path = tmp_path / "prepare.json"
    selected_quality_shard_path = tmp_path / "quality_shard.json"
    pw04_summary_path = tmp_path / "pw04_summary.json"

    assert resolve_pw04_expected_output(
        pw04_mode="prepare",
        prepare_manifest_path=prepare_manifest_path,
        selected_quality_shard_path=selected_quality_shard_path,
        pw04_summary_path=pw04_summary_path,
    ) == ("prepare_manifest", prepare_manifest_path)
    assert resolve_pw04_expected_output(
        pw04_mode="quality_shard",
        prepare_manifest_path=prepare_manifest_path,
        selected_quality_shard_path=selected_quality_shard_path,
        pw04_summary_path=pw04_summary_path,
    ) == ("quality_shard", selected_quality_shard_path)
    assert resolve_pw04_expected_output(
        pw04_mode="finalize",
        prepare_manifest_path=prepare_manifest_path,
        selected_quality_shard_path=selected_quality_shard_path,
        pw04_summary_path=pw04_summary_path,
    ) == ("pw04_summary", pw04_summary_path)


def test_load_gpu_peak_summary_and_build_notebook_summary(tmp_path: Path) -> None:
    """
    Verify GPU peak summary helpers cover missing and valid payloads.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    missing_payload, missing_error = load_gpu_peak_summary(tmp_path / "missing.json")
    assert missing_payload is None
    assert isinstance(missing_error, str)

    summary_path = tmp_path / "gpu_session_peak.json"
    _write_json(
        summary_path,
        {
            "status": "completed",
            "started_at_utc": "2026-04-20T00:00:00+00:00",
            "finished_at_utc": "2026-04-20T00:00:03+00:00",
            "elapsed_seconds": 3.0,
            "session_board_peak_memory_used_mib": 2048,
            "session_board_peak_increment_mib": 1024,
            "peak_gpu_index": 0,
            "peak_gpu_uuid": "GPU-FAKE",
            "peak_gpu_name": "Fake GPU",
            "visible_gpu_count": 1,
            "wrapped_command": ["python", "demo.py"],
            "wrapped_return_code": 0,
        },
    )
    payload, error_text = load_gpu_peak_summary(summary_path)

    assert error_text is None
    notebook_summary = build_gpu_peak_notebook_summary(
        raw_summary=payload,
        monitor_status="completed",
        monitor_error=None,
        fallback_reason=None,
        display_helper=lambda summary: {
            "status": summary.get("status"),
            "peak_memory_used_mib": summary.get("session_board_peak_memory_used_mib"),
            "peak_memory_used_gib": 2.0,
            "start_memory_used_mib": None,
            "end_memory_used_mib": None,
            "peak_gpu_name": summary.get("peak_gpu_name"),
            "peak_gpu_index": summary.get("peak_gpu_index"),
            "visible_gpu_count": summary.get("visible_gpu_count"),
            "recommendation": "24 GB 档更稳妥",
        },
        gpu_peak_summary_path=summary_path,
    )

    assert notebook_summary["gpu_session_peak_path"] == str(summary_path)
    assert notebook_summary["gpu_peak_memory_mib"] == 2048
    assert notebook_summary["peak_gpu_name"] == "Fake GPU"
    assert notebook_summary["monitor_started_at_utc"] == "2026-04-20T00:00:00+00:00"
    assert notebook_summary["monitor_finished_at_utc"] == "2026-04-20T00:00:03+00:00"
    assert notebook_summary["monitor_elapsed_seconds"] == 3.0
    assert notebook_summary["wrapped_command"] == ["python", "demo.py"]
    assert notebook_summary["monitor_recommendation"] == "24 GB 档更稳妥"


def test_read_pw04_result_summary_prepare_reads_manifest_bound_artifacts(tmp_path: Path) -> None:
    """
    Verify prepare mode reads all required manifest-bound artifacts.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    family_root = tmp_path / "family"
    exports_root = family_root / "exports" / "pw04"
    prepare_manifest_path = exports_root / "manifests" / "pw04_prepare_manifest.json"
    selected_quality_shard_path = exports_root / "quality" / "shards" / "quality_shard_0000.json"
    pw04_summary_path = family_root / "runtime_state" / "pw04_summary.json"

    attack_merge_manifest_path = exports_root / "manifests" / "attack_merge_manifest.json"
    attack_positive_pool_manifest_path = exports_root / "attack_positive_pool_manifest.json"
    attack_negative_pool_manifest_path = exports_root / "attack_negative_pool_manifest.json"
    formal_attack_final_decision_metrics_path = exports_root / "formal_attack_final_decision_metrics.json"
    formal_attack_attestation_metrics_path = exports_root / "formal_attack_attestation_metrics.json"
    derived_attack_union_metrics_path = exports_root / "derived_attack_union_metrics.json"
    formal_attack_negative_metrics_path = exports_root / "formal_attack_negative_metrics.json"
    quality_pair_plan_path = exports_root / "quality" / "quality_pair_plan.json"

    _write_json(attack_merge_manifest_path, {"artifact_type": "attack_merge_manifest"})
    _write_json(attack_positive_pool_manifest_path, {"artifact_type": "attack_positive_pool_manifest"})
    _write_json(attack_negative_pool_manifest_path, {"artifact_type": "attack_negative_pool_manifest"})
    _write_json(formal_attack_final_decision_metrics_path, {"scope": "formal_final"})
    _write_json(formal_attack_attestation_metrics_path, {"scope": "attestation"})
    _write_json(derived_attack_union_metrics_path, {"scope": "derived_union"})
    _write_json(formal_attack_negative_metrics_path, {"scope": "formal_negative"})
    _write_json(quality_pair_plan_path, {"quality_shard_count": 1})
    _write_json(selected_quality_shard_path, {"artifact_type": "quality_shard"})
    _write_json(
        prepare_manifest_path,
        {
            "attack_merge_manifest_path": str(attack_merge_manifest_path),
            "attack_positive_pool_manifest_path": str(attack_positive_pool_manifest_path),
            "attack_negative_pool_manifest_path": str(attack_negative_pool_manifest_path),
            "formal_attack_final_decision_metrics_path": str(formal_attack_final_decision_metrics_path),
            "formal_attack_attestation_metrics_path": str(formal_attack_attestation_metrics_path),
            "derived_attack_union_metrics_path": str(derived_attack_union_metrics_path),
            "formal_attack_negative_metrics_path": str(formal_attack_negative_metrics_path),
            "quality_pair_plan_path": str(quality_pair_plan_path),
            "expected_quality_shard_paths": [str(selected_quality_shard_path)],
        },
    )

    summary = read_pw04_result_summary(
        pw04_mode="prepare",
        family_root=family_root,
        prepare_manifest_path=prepare_manifest_path,
        selected_quality_shard_path=selected_quality_shard_path,
        pw04_summary_path=pw04_summary_path,
        gpu_peak_notebook_summary={"monitor_status": "completed"},
    )

    assert summary["mode"] == "prepare"
    assert summary["attack_merge_manifest"]["artifact_type"] == "attack_merge_manifest"
    assert summary["quality_pair_plan"]["quality_shard_count"] == 1
    assert summary["expected_quality_shard_paths"] == [
        {"path": str(selected_quality_shard_path), "exists": True}
    ]


def test_read_pw04_result_summary_quality_shard_reads_selected_shard(tmp_path: Path) -> None:
    """
    Verify quality-shard mode reads the selected shard and frozen prepare manifest.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    family_root = tmp_path / "family"
    prepare_manifest_path = family_root / "exports" / "pw04" / "manifests" / "pw04_prepare_manifest.json"
    selected_quality_shard_path = family_root / "exports" / "pw04" / "quality" / "shards" / "quality_shard_0000.json"
    pw04_summary_path = family_root / "runtime_state" / "pw04_summary.json"

    _write_json(
        prepare_manifest_path,
        {
            "expected_quality_shard_paths": [str(selected_quality_shard_path)],
            "quality_pair_plan_path": str(family_root / "exports" / "pw04" / "quality" / "quality_pair_plan.json"),
        },
    )
    _write_json(selected_quality_shard_path, {"quality_shard_index": 0, "attack_pair_count": 2})

    summary = read_pw04_result_summary(
        pw04_mode="quality_shard",
        family_root=family_root,
        prepare_manifest_path=prepare_manifest_path,
        selected_quality_shard_path=selected_quality_shard_path,
        pw04_summary_path=pw04_summary_path,
        gpu_peak_notebook_summary={"monitor_status": "completed"},
    )

    assert summary["mode"] == "quality_shard"
    assert summary["quality_shard"]["quality_shard_index"] == 0
    assert summary["quality_shard"]["attack_pair_count"] == 2


def test_read_pw04_result_summary_finalize_reads_summary_and_exports(tmp_path: Path) -> None:
    """
    Verify finalize mode reads summary, canonical metrics, and paper export bindings.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """
    family_root = tmp_path / "family"
    exports_root = family_root / "exports" / "pw04"
    runtime_root = family_root / "runtime_state"
    prepare_manifest_path = exports_root / "manifests" / "pw04_prepare_manifest.json"
    selected_quality_shard_path = exports_root / "quality" / "shards" / "quality_shard_0000.json"
    pw04_summary_path = runtime_root / "pw04_summary.json"

    formal_attack_final_decision_metrics_path = exports_root / "formal_attack_final_decision_metrics.json"
    formal_attack_attestation_metrics_path = exports_root / "formal_attack_attestation_metrics.json"
    derived_attack_union_metrics_path = exports_root / "derived_attack_union_metrics.json"
    clean_attack_overview_path = exports_root / "clean_attack_overview.json"
    paper_metric_registry_path = exports_root / "metrics" / "paper_metric_registry.json"
    content_chain_metrics_path = exports_root / "metrics" / "content_chain_metrics.json"
    event_attestation_metrics_path = exports_root / "metrics" / "event_attestation_metrics.json"
    system_final_metrics_path = exports_root / "metrics" / "system_final_metrics.json"
    bootstrap_confidence_intervals_path = exports_root / "metrics" / "bootstrap_confidence_intervals.json"
    bootstrap_confidence_intervals_csv_path = exports_root / "metrics" / "bootstrap_confidence_intervals.csv"
    figure_path = exports_root / "figures" / "attack_tpr_by_family.png"
    tail_fpr_1e4_path = exports_root / "tail" / "estimated_tail_fpr_1e4.json"
    tail_fpr_1e5_path = exports_root / "tail" / "estimated_tail_fpr_1e5.json"
    tail_fit_diagnostics_path = exports_root / "tail" / "tail_fit_diagnostics.json"
    tail_fit_stability_summary_path = exports_root / "tail" / "tail_fit_stability_summary.json"

    _write_json(formal_attack_final_decision_metrics_path, {"scope": "formal_final"})
    _write_json(formal_attack_attestation_metrics_path, {"scope": "attestation"})
    _write_json(derived_attack_union_metrics_path, {"scope": "derived_union"})
    _write_json(clean_attack_overview_path, {"scope": "clean_vs_attack"})
    _write_json(paper_metric_registry_path, {"artifact_type": "paper_metric_registry"})
    _write_json(content_chain_metrics_path, {"scope": "content_chain"})
    _write_json(event_attestation_metrics_path, {"scope": "event_attestation"})
    _write_json(system_final_metrics_path, {"scope": "system_final"})
    _write_json(bootstrap_confidence_intervals_path, {"artifact_type": "bootstrap_confidence_intervals"})
    _write_text(bootstrap_confidence_intervals_csv_path, "scope,metric_name\n")
    _write_text(figure_path, "figure")
    _write_json(tail_fpr_1e4_path, {"target": "1e-4"})
    _write_json(tail_fpr_1e5_path, {"target": "1e-5"})
    _write_json(tail_fit_diagnostics_path, {"artifact_type": "tail_fit_diagnostics"})
    _write_json(tail_fit_stability_summary_path, {"artifact_type": "tail_fit_stability_summary"})
    _write_json(
        pw04_summary_path,
        {
            "status": "completed",
            "paper_scope_registry_path": str(paper_metric_registry_path),
            "canonical_metrics_paths": {
                "content_chain": str(content_chain_metrics_path),
                "event_attestation": str(event_attestation_metrics_path),
                "system_final": str(system_final_metrics_path),
            },
            "paper_tables_paths": {
                "main_metrics_summary_csv_path": str(exports_root / "tables" / "main_metrics_summary.csv"),
            },
            "paper_figures_paths": {
                "attack_tpr_by_family_figure_path": str(figure_path),
            },
            "tail_estimation_paths": {
                "estimated_tail_fpr_1e4_path": str(tail_fpr_1e4_path),
                "estimated_tail_fpr_1e5_path": str(tail_fpr_1e5_path),
                "tail_fit_diagnostics_path": str(tail_fit_diagnostics_path),
                "tail_fit_stability_summary_path": str(tail_fit_stability_summary_path),
            },
            "bootstrap_confidence_intervals_path": str(bootstrap_confidence_intervals_path),
            "bootstrap_confidence_intervals_csv_path": str(bootstrap_confidence_intervals_csv_path),
        },
    )

    summary = read_pw04_result_summary(
        pw04_mode="finalize",
        family_root=family_root,
        prepare_manifest_path=prepare_manifest_path,
        selected_quality_shard_path=selected_quality_shard_path,
        pw04_summary_path=pw04_summary_path,
        gpu_peak_notebook_summary={"monitor_status": "completed", "gpu_session_peak_path": "gpu.json"},
    )

    assert summary["mode"] == "finalize"
    assert summary["summary"]["status"] == "completed"
    assert summary["paper_metric_registry"]["artifact_type"] == "paper_metric_registry"
    assert summary["canonical_metrics_paths"]["content_chain"] == str(content_chain_metrics_path)
    assert summary["content_chain_metrics"]["scope"] == "content_chain"
    assert summary["paper_figures_paths"]["attack_tpr_by_family_figure_path"] == {
        "path": str(figure_path),
        "exists": True,
    }
    assert summary["tail_estimation_paths"]["estimated_tail_fpr_1e4_path"] == str(tail_fpr_1e4_path)
    assert summary["bootstrap_confidence_intervals"]["artifact_type"] == "bootstrap_confidence_intervals"