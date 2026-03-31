"""
文件目的：验证 canonical run_closure 会保留命令级 closure 历史视图。
Module type: General module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

import pytest

from main.core import status
from main.core.errors import RunFailureReason


def _build_runtime_finalization_details(phase_label: str) -> Dict[str, Any]:
    """
    功能：构造测试用 runtime finalization 细节。

    Build a minimal runtime-finalization detail mapping for closure history tests.

    Args:
        phase_label: CUDA memory profile phase label.

    Returns:
        Runtime-finalization detail mapping.
    """
    return {
        "runtime_finalization_status": "ok",
        "runtime_executable_plan_status": "ok",
        "runtime_capture_cuda_memory_profile": {
            "status": "absent",
            "reason": "cuda_not_active",
            "phase_label": phase_label,
            "sample_scope": "single_worker_process_local",
            "device": "cpu",
        },
    }


def _build_run_meta(command: str, run_id: str, phase_label: str) -> Dict[str, Any]:
    """
    功能：构造最小失败路径 run_meta。

    Build the minimal failure-path run_meta used by finalize_run history tests.

    Args:
        command: Command name.
        run_id: Run identifier.
        phase_label: CUDA memory profile phase label.

    Returns:
        Run metadata mapping.
    """
    return {
        "run_id": run_id,
        "command": command,
        "created_at_utc": "2026-03-31T00:00:00.000Z",
        "started_at": "2026-03-31T00:00:00.000Z",
        "ended_at": "2026-03-31T00:00:01.000Z",
        "cfg_digest": f"cfg_digest_{command}",
        "policy_path": "content_np_geo_rescue",
        "impl_id": "impl_anchor",
        "impl_version": "v1",
        "status_ok": False,
        "status_reason": RunFailureReason.RUNTIME_ERROR,
        "status_details": {
            "runtime_finalization": _build_runtime_finalization_details(phase_label),
        },
    }


def test_finalize_run_preserves_command_closure_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：同一 run_root 多次 finalize 时必须保留 embed 与 detect 的 command 视图。

    Verify finalize_run preserves command-scoped closure views when multiple
    commands write the same canonical run_closure.json.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    run_root = tmp_path / "run"
    records_dir = run_root / "records"
    artifacts_dir = run_root / "artifacts"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _write_artifact_json_bound(
        _run_root: Path,
        _records_dir: Path,
        _artifacts_dir: Path,
        _logs_dir: Path,
        path_obj: Path,
        payload: Dict[str, Any],
    ) -> None:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(status.path_policy, "validate_output_target", cast(Any, lambda *args, **kwargs: None))
    monkeypatch.setattr(status, "_write_artifact_json_bound", _write_artifact_json_bound)

    embed_meta = _build_run_meta("embed", "run-embed", "statement_only_runtime_capture")
    detect_meta = _build_run_meta("detect", "run-detect", "detect_main_inference")

    status.finalize_run(run_root, records_dir, artifacts_dir, embed_meta)
    run_closure_path = status.finalize_run(run_root, records_dir, artifacts_dir, detect_meta)
    run_closure = json.loads(run_closure_path.read_text(encoding="utf-8"))

    assert run_closure["command"] == "detect"
    assert run_closure["status"]["reason"] == RunFailureReason.RUNTIME_ERROR.value
    assert run_closure["status"]["details"]["runtime_finalization"]["runtime_capture_cuda_memory_profile"][
        "phase_label"
    ] == "detect_main_inference"

    command_closures = run_closure["command_closures"]
    assert command_closures["embed"]["command"] == "embed"
    assert command_closures["detect"]["command"] == "detect"
    assert command_closures["embed"]["status"]["details"]["runtime_finalization"][
        "runtime_capture_cuda_memory_profile"
    ]["phase_label"] == "statement_only_runtime_capture"
    assert command_closures["detect"]["status"]["details"]["runtime_finalization"][
        "runtime_capture_cuda_memory_profile"
    ]["phase_label"] == "detect_main_inference"