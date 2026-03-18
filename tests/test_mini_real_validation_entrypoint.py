"""
文件目的：mini real validation 入口脚本回归测试。
Module type: General module

覆盖点：
1. 默认入口必须绑定唯一 mini-real 配置与输出目录。
2. 失败摘要必须结构化暴露 environment、run_closure 与 experiment_matrix 失败原因。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module(module_name: str, module_path: Path) -> Any:
    """
    功能：按路径动态加载模块。

    Dynamically load a module from a file path.

    Args:
        module_name: Import name used for the dynamic module.
        module_path: Target Python file path.

    Returns:
        Imported module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_mini_real_entrypoint_defaults_to_mini_real_profile() -> None:
    """
    功能：mini real 验收入口必须绑定唯一 mini-real 配置与输出目录。

    Verify the mini-real entrypoint binds the dedicated config and run root.

    Args:
        None.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_mini_real_validation_test",
        repo_root / "scripts" / "run_mini_real_validation.py",
    )

    assert module.DEFAULT_CONFIG_PATH.as_posix() == "configs/paper_full_cuda_mini_real_validation.yaml"
    assert module.DEFAULT_RUN_ROOT.as_posix() == "outputs/onefile_paper_full_cuda_mini_real_validation"


def test_run_mini_real_validation_emits_structured_failure_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：mini real 失败摘要必须结构化暴露 matrix 与 run_closure 主失败原因。

    Verify run_mini_real_validation returns a structured failure summary with
    experiment_matrix and run_closure evidence.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_mini_real_validation_summary_test",
        repo_root / "scripts" / "run_mini_real_validation.py",
    )

    run_root = tmp_path / "mini_real_run"
    config_path = tmp_path / "mini_real.yaml"
    config_path.write_text(
        "\n".join([
            'device: "cuda"',
            "experiment_matrix:",
            "  require_real_negative_cache: true",
            "  require_shared_thresholds: true",
            "  disallow_forced_pair_fallback: true",
        ]),
        encoding="utf-8",
    )

    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
    (run_root / "outputs" / "experiment_matrix" / "artifacts").mkdir(parents=True, exist_ok=True)

    (run_root / "records" / "embed_record.json").write_text("{}", encoding="utf-8")
    (run_root / "records" / "detect_record.json").write_text(
        json.dumps({
            "content_evidence_payload": {
                "status": "mismatch",
                "content_failure_reason": "mask_extraction_no_input",
            }
        }),
        encoding="utf-8",
    )
    (run_root / "records" / "calibration_record.json").write_text("{}", encoding="utf-8")
    (run_root / "records" / "evaluate_record.json").write_text("{}", encoding="utf-8")
    (run_root / "artifacts" / "evaluation_report.json").write_text("{}", encoding="utf-8")
    (run_root / "artifacts" / "run_closure.json").write_text(
        json.dumps({
            "pipeline_build_failure_reason": "<absent>",
            "status": {
                "ok": False,
                "reason": "runtime_error",
                "details": {
                    "upstream_failure_reason": "experiment_matrix_failed",
                },
            },
        }),
        encoding="utf-8",
    )
    (run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json").write_text(
        json.dumps({
            "total": 2,
            "failed": 2,
            "results": [
                {
                    "status": "failed",
                    "failure_reason": "experiment_matrix formal validation disallows forced pair fallback",
                }
            ],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "detect_formal_gpu_preflight", lambda _cfg: {"ok": True, "gpu_tool_available": True, "missing_attestation_env_vars": []})
    monkeypatch.setattr(module, "run_onefile_workflow", lambda **kwargs: 1)

    result = module.run_mini_real_validation(run_root=run_root, config_path=config_path, dry_run=False)
    summary = result["summary"]

    assert result["workflow_exit_code"] == 1
    assert summary["profile_role"] == "paper_full_cuda_mini_real_validation"
    assert summary["mini_real_validation_ok"] is False
    assert summary["nearest_failed_stage"] == "experiment_matrix"
    assert summary["formal_requirements"]["require_real_negative_cache"] is True
    assert summary["formal_requirements"]["require_shared_thresholds"] is True
    assert summary["formal_requirements"]["disallow_forced_pair_fallback"] is True
    assert summary["issue_count"] >= 2
    issue_names = {item["issue"] for item in summary["issues"]}
    assert "run_closure_failed" in issue_names
    assert "experiment_matrix_failed" in issue_names
    assert summary["details"]["matrix_first_failure_reason"] == (
        "experiment_matrix formal validation disallows forced pair fallback"
    )
    assert Path(result["summary_path"]).exists()


def test_run_mini_real_validation_reads_matrix_failure_from_workflow_log_when_summary_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """
    功能：当 grid_summary 缺失时，入口摘要必须回填 workflow log 中的 matrix 主失败原因。

    Verify run_mini_real_validation surfaces the experiment_matrix failure from
    the workflow log when grid_summary.json is absent.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_module(
        "run_mini_real_validation_log_backfill_test",
        repo_root / "scripts" / "run_mini_real_validation.py",
    )

    run_root = tmp_path / "mini_real_run"
    config_path = tmp_path / "mini_real.yaml"
    config_path.write_text(
        "\n".join([
            'device: "cuda"',
            "experiment_matrix:",
            "  require_real_negative_cache: true",
            "  require_shared_thresholds: true",
            "  disallow_forced_pair_fallback: true",
        ]),
        encoding="utf-8",
    )

    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)

    (run_root / "records" / "embed_record.json").write_text("{}", encoding="utf-8")
    (run_root / "records" / "detect_record.json").write_text("{}", encoding="utf-8")
    (run_root / "records" / "calibration_record.json").write_text("{}", encoding="utf-8")
    (run_root / "records" / "evaluate_record.json").write_text("{}", encoding="utf-8")
    (run_root / "artifacts" / "evaluation_report.json").write_text("{}", encoding="utf-8")
    (run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").write_text("{}", encoding="utf-8")
    (run_root / "artifacts" / "run_closure.json").write_text(
        json.dumps({
            "pipeline_build_failure_reason": "<absent>",
            "status": {
                "ok": True,
                "reason": "ok",
                "details": None,
            },
        }),
        encoding="utf-8",
    )
    (run_root / "logs" / "mini_real_workflow_execution.log").write_text(
        "\n".join([
            "[onefile] step=experiment_matrix start=2026-03-18 04:53:04",
            "[ExperimentMatrixBatch] [ERROR] RuntimeError: experiment_matrix formal validation requires real negative cache for every guarded item; missing=model_id=stabilityai/stable-diffusion-3.5-medium, seed=0; model_id=stabilityai/stable-diffusion-3.5-medium, seed=1",
            "Traceback (most recent call last):",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "detect_formal_gpu_preflight", lambda _cfg: {"ok": True, "gpu_tool_available": True, "missing_attestation_env_vars": []})
    monkeypatch.setattr(module, "run_onefile_workflow", lambda **kwargs: 1)

    result = module.run_mini_real_validation(run_root=run_root, config_path=config_path, dry_run=False)
    summary = result["summary"]

    assert result["workflow_exit_code"] == 1
    assert summary["nearest_failed_stage"] == "experiment_matrix"
    assert summary["details"]["matrix_log_failure_reason"] == (
        "RuntimeError: experiment_matrix formal validation requires real negative cache for every guarded item; missing=model_id=stabilityai/stable-diffusion-3.5-medium, seed=0; model_id=stabilityai/stable-diffusion-3.5-medium, seed=1"
    )
    matrix_issues = [item for item in summary["issues"] if item["issue"] == "experiment_matrix_failed_before_summary_write"]
    assert len(matrix_issues) == 1
    assert matrix_issues[0]["evidence"]["first_failure_reason"].startswith("RuntimeError: experiment_matrix formal validation requires real negative cache")