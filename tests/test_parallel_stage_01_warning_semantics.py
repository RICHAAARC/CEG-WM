"""
文件目的：验证 parallel runner 对 direct_stats_ready / final_label_balanced 的 shared warning 语义不做重新解释。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PARALLEL_RUNNER_PATH = REPO_ROOT / "scripts" / "01_run_paper_full_cuda_parallel.py"
PARALLEL_WORKER_PATH = REPO_ROOT / "scripts" / "01_run_paper_full_cuda_parallel_worker.py"
BASE_RUNNER_PATH = REPO_ROOT / "scripts" / "01_run_paper_full_cuda.py"


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    """
    功能：按文件路径加载测试目标模块。

    Load one target module from a file path.

    Args:
        module_path: Module file path.
        module_name: Synthetic module name.

    Returns:
        Loaded module object.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARALLEL_RUNNER = _load_module(PARALLEL_RUNNER_PATH, "test_parallel_stage_01_warning_semantics_runner")
PARALLEL_WORKER = _load_module(PARALLEL_WORKER_PATH, "test_parallel_stage_01_warning_semantics_worker")


def _write_json(path_obj: Path, payload: Dict[str, Any]) -> None:
    """
    功能：写入 JSON 测试工件。

    Write a JSON payload for tests.

    Args:
        path_obj: Destination path.
        payload: JSON-serializable mapping.

    Returns:
        None.
    """
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_parallel_runner_preserves_shared_stage_02_warning_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：parallel runner 必须允许 direct_stats_ready=false 与 final_label_balanced=true 并存，不得把该 shared warning 语义提升为 blocker。

    Verify that the parallel runner preserves the shared warning semantics for
    direct_stats_ready and final_label_balanced without promoting them to a
    blocking failure.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    shared_runner = PARALLEL_RUNNER.BASE_RUNNER_MODULE
    assert Path(shared_runner.__file__).resolve() == BASE_RUNNER_PATH.resolve()

    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    run_root = tmp_path / "parallel_run_root"

    representative_embed_record_path = run_root / shared_runner.SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT / "000_embed_record.json"
    representative_detect_record_path = run_root / shared_runner.SOURCE_POOL_DETECT_RECORDS_RELATIVE_ROOT / "000_detect_record.json"
    _write_json(representative_embed_record_path, {"content_evidence": {"audit": {}}})
    _write_json(representative_detect_record_path, {"label": True})

    direct_entry = {
        "prompt_index": 0,
        "prompt_text": "parallel prompt 0",
        "path": representative_detect_record_path.as_posix(),
        "package_relative_path": f"{shared_runner.SOURCE_POOL_DETECT_RECORDS_RELATIVE_ROOT}/000_detect_record.json",
        "stage_results": {"embed": {"return_code": 0}, "detect": {"return_code": 0}},
    }
    worker_result = PARALLEL_WORKER._build_worker_result_payload(
        stage_run_id="parallel_warning_stage",
        prompt_file_path="prompts/paper_small.txt",
        worker_index=0,
        worker_count=1,
        assigned_prompt_indices=[0],
        direct_entries=[direct_entry],
        source_pool_stage_results=[
            {
                "prompt_index": 0,
                "prompt_text": "parallel prompt 0",
                "stage_results": direct_entry["stage_results"],
                "package_relative_path": direct_entry["package_relative_path"],
            }
        ],
        status="ok",
    )

    monkeypatch.setattr(
        PARALLEL_RUNNER,
        "load_yaml_mapping",
        lambda _path: {
            "parallel_attestation_statistics": {"enabled": True},
            "stage_01_source_pool": {"enabled": True},
            "stage_01_pooled_threshold_build": {"enabled": True},
        },
    )
    monkeypatch.setattr(shared_runner, "_resolve_stage_01_source_pool_cfg", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(shared_runner, "_resolve_stage_01_pooled_threshold_build_cfg", lambda _cfg: {"enabled": True})
    monkeypatch.setattr(shared_runner, "_resolve_stage_01_prompt_pool", lambda _cfg: (["parallel prompt 0"], "prompts/paper_small.txt"))
    monkeypatch.setattr(PARALLEL_RUNNER, "_run_parallel_workers", lambda **_kwargs: ([], [worker_result]))
    monkeypatch.setattr(
        shared_runner,
        "_build_stage_01_canonical_source_pool",
        lambda **_kwargs: {"manifest_package_relative_path": "artifacts/stage_01_canonical_source_pool/source_pool_manifest.json"},
    )
    monkeypatch.setattr(
        shared_runner,
        "_build_stage_01_source_contract",
        lambda **_kwargs: {
            "record_count": 1,
            "direct_stats_ready": False,
            "direct_stats_reason": "parallel_attestation_statistics_requires_label_balanced_detect_records",
        },
    )
    monkeypatch.setattr(
        shared_runner,
        "_build_stage_01_pooled_threshold_records",
        lambda **_kwargs: {"final_label_balanced": True},
    )
    monkeypatch.setattr(shared_runner, "_build_pooled_runtime_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(shared_runner, "_build_stage_command", lambda *_args, **_kwargs: ["noop"])
    monkeypatch.setattr(shared_runner, "_run_stage", lambda *_args, **_kwargs: {"return_code": 0})
    monkeypatch.setattr(shared_runner, "_artifact_presence", lambda _required: {})
    monkeypatch.setattr(shared_runner, "_required_artifacts", lambda _run_root: [])
    monkeypatch.setattr(shared_runner, "_all_required_present", lambda _artifact_summary: True)
    monkeypatch.setattr(shared_runner, "_resolve_stage_01_attestation_evidence", lambda **_kwargs: {"overall_status": "ok"})

    exit_code = PARALLEL_RUNNER.run_paper_full_cuda_parallel(
        config_path,
        run_root,
        stage_run_id="parallel_warning_stage",
        worker_count=1,
    )

    source_contract = json.loads((run_root / shared_runner.SOURCE_CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    pooled_build_contract = json.loads(
        (run_root / shared_runner.POOLED_THRESHOLD_BUILD_CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    workflow_summary = json.loads((run_root / "artifacts" / "workflow_summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert source_contract["direct_stats_ready"] is False
    assert source_contract["direct_stats_reason"] == "parallel_attestation_statistics_requires_label_balanced_detect_records"
    assert pooled_build_contract["final_label_balanced"] is True
    assert workflow_summary["status"] == "ok"