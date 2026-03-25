"""
File purpose: output-only paper_full_cuda 编排回归测试。
Module type: General module
"""

from __future__ import annotations

from pathlib import Path

from scripts import run_paper_full_cuda as module


def test_run_paper_full_cuda_detaches_parallel_attestation_statistics_when_enabled(monkeypatch) -> None:
    """
    功能：启用 parallel_attestation_statistics 时主脚本只保留 detached post-flow 说明。

    Verify the output-only paper_full_cuda workflow no longer runs the
    parallel attestation statistics subflow inside the main workflow.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    emitted_summaries = []

    monkeypatch.setattr(
        module,
        "_load_runtime_config",
        lambda _path: {
            "attestation": {
                "enabled": True,
            },
            "parallel_attestation_statistics": {
                "enabled": True,
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            }
        },
    )
    monkeypatch.setattr(module, "_resolve_main_attestation_evidence", lambda _cfg, _run_root: {"enabled": True, "status": "ok"})
    monkeypatch.setattr(module, "_persist_workflow_summary", lambda _run_root, _summary: None)
    monkeypatch.setattr(module, "_emit_workflow_summary", lambda summary: emitted_summaries.append(summary))

    def _fake_run_step(step_name, command):
        _ = command
        emitted_summaries.append({"step_name": step_name})
        return 0

    monkeypatch.setattr(module, "_run_step", _fake_run_step)

    result = module.run_paper_full_cuda(Path("configs/paper_full_cuda.yaml"), Path("outputs/run"))

    assert result == 0
    stage_order = [item["step_name"] for item in emitted_summaries if "step_name" in item]
    assert stage_order == [
        "embed",
        "detect",
        "calibrate",
        "evaluate",
    ]
    summary = next(item for item in emitted_summaries if "main_status" in item)
    assert summary["workflow_status"] == "ok"
    assert summary["parallel_attestation_statistics"]["status"] == "detached_not_run"
    assert summary["parallel_attestation_statistics"]["execution_mode"] == "independent_post_flow"
    assert summary["parallel_attestation_statistics"]["script_path"] == "scripts/run_parallel_attestation_statistics.py"


def test_run_paper_full_cuda_detached_parallel_statistics_do_not_override_main_success(monkeypatch) -> None:
    """
    功能：parallel_attestation_statistics 从主脚本剥离后不得反向污染主链成功判定。

    Verify detached parallel attestation metadata does not override main-chain success.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    emitted_summaries = []

    monkeypatch.setattr(
        module,
        "_load_runtime_config",
        lambda _path: {
            "attestation": {
                "enabled": True,
            },
            "parallel_attestation_statistics": {
                "enabled": True,
                "calibration_score_name": "event_attestation_score",
                "evaluate_score_name": "event_attestation_score",
            }
        },
    )
    monkeypatch.setattr(module, "_resolve_main_attestation_evidence", lambda _cfg, _run_root: {"enabled": True, "status": "ok"})
    monkeypatch.setattr(module, "_persist_workflow_summary", lambda _run_root, _summary: None)
    monkeypatch.setattr(module, "_emit_workflow_summary", lambda summary: emitted_summaries.append(summary))
    monkeypatch.setattr(module, "_run_step", lambda _step_name, _command: 0)

    result = module.run_paper_full_cuda(Path("configs/paper_full_cuda.yaml"), Path("outputs/run"))

    assert result == 0
    assert len(emitted_summaries) == 1
    summary = emitted_summaries[0]
    assert summary["main_status"] == "ok"
    assert summary["main_chain"]["status"] == "ok"
    assert summary["workflow_status"] == "ok"
    assert summary["parallel_attestation_statistics"]["status"] == "detached_not_run"


def test_run_paper_full_cuda_matrix_failure_does_not_override_main_success(monkeypatch) -> None:
    """
    功能：experiment_matrix 失败不得抹杀主四阶段已成功产出的项目输出。

    Verify experiment_matrix failure remains non-blocking once the main output
    stages have completed successfully.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        module,
        "_load_runtime_config",
        lambda _path: {
            "attestation": {
                "enabled": True,
            },
            "experiment_matrix": {
                "config_path": "configs/paper_full_cuda.yaml",
            }
        },
    )
    monkeypatch.setattr(module, "_resolve_main_attestation_evidence", lambda _cfg, _run_root: {"enabled": True, "status": "ok"})
    monkeypatch.setattr(module, "_persist_workflow_summary", lambda _run_root, _summary: None)
    monkeypatch.setattr(module, "_emit_workflow_summary", lambda _summary: None)

    def _fake_run_step(step_name, command):
        _ = command
        if step_name == "experiment_matrix":
            return 2
        return 0

    monkeypatch.setattr(module, "_run_step", _fake_run_step)

    result = module.run_paper_full_cuda(Path("configs/paper_full_cuda.yaml"), Path("outputs/run"))

    assert result == 0


def test_run_paper_full_cuda_requires_main_attestation_evidence(monkeypatch) -> None:
    """
    功能：主链若缺失 attestation 真实参与证据，workflow 不得判定为成功。

    Verify main-chain success is blocked when attestation evidence is missing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    emitted_summaries = []

    monkeypatch.setattr(
        module,
        "_load_runtime_config",
        lambda _path: {
            "attestation": {
                "enabled": True,
            }
        },
    )
    monkeypatch.setattr(
        module,
        "_resolve_main_attestation_evidence",
        lambda _cfg, _run_root: {
            "enabled": True,
            "status": "missing_evidence",
            "missing": ["attestation_result.final_event_attested_decision.event_attestation_score"],
        },
    )
    monkeypatch.setattr(module, "_persist_workflow_summary", lambda _run_root, _summary: None)
    monkeypatch.setattr(module, "_emit_workflow_summary", lambda summary: emitted_summaries.append(summary))
    monkeypatch.setattr(module, "_run_step", lambda _step_name, _command: 0)

    result = module.run_paper_full_cuda(Path("configs/paper_full_cuda.yaml"), Path("outputs/run"))

    assert result == 1
    assert len(emitted_summaries) == 1
    summary = emitted_summaries[0]
    assert summary["main_status"] == "failed_attestation_evidence"
    assert summary["main_chain"]["status"] == "failed_attestation_evidence"
    assert summary["workflow_status"] == "failed"