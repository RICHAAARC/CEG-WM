"""
文件目的：验证 parallel worker 通过 shared baseline preview writer 继承 preview_generation phase label 修复。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
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


PARALLEL_WORKER = _load_module(PARALLEL_WORKER_PATH, "test_parallel_stage_01_preview_phase_label_worker")


def test_parallel_worker_shared_preview_writer_persists_preview_generation_phase_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：parallel worker 通过 shared baseline preview writer 生成的 preview record 必须保留 preview_generation phase label。

    Verify that the preview-generation record reached through the parallel
    worker's shared baseline runner persists the preview_generation phase label.

    Args:
        tmp_path: Temporary pytest directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    shared_runner = PARALLEL_WORKER.BASE_RUNNER_MODULE
    assert Path(shared_runner.__file__).resolve() == BASE_RUNNER_PATH.resolve()

    class _FakePreviewImage:
        def save(self, path_value: str | Path) -> None:
            Path(path_value).write_bytes(b"preview")

    captured_inference_kwargs: Dict[str, Any] = {}

    def _fake_run_sd3_inference(*_args: Any, **kwargs: Any) -> Dict[str, Any]:
        captured_inference_kwargs.update(kwargs)
        return {
            "inference_status": shared_runner.infer_runtime.INFERENCE_STATUS_OK,
            "inference_error": None,
            "inference_runtime_meta": {
                "latency_ms": 1.0,
                "cuda_memory_profile": {
                    "status": "absent",
                    "reason": "cuda_not_active",
                    "phase_label": kwargs.get("runtime_phase_label"),
                    "sample_scope": "single_worker_process_local",
                    "device": "cpu",
                },
            },
            "output_image": _FakePreviewImage(),
        }

    monkeypatch.setattr(shared_runner, "build_seed_audit", lambda *_args, **_kwargs: ({}, "seed_digest", 7, "seed_rule"))
    monkeypatch.setattr(
        shared_runner.pipeline_factory,
        "build_pipeline_shell",
        lambda _cfg: {
            "pipeline_obj": object(),
            "pipeline_status": "built",
            "pipeline_error": None,
            "pipeline_runtime_meta": {
                "model_source_resolution": "local_snapshot_priority",
                "local_snapshot_status": "bound",
                "resolved_model_source": "local_path",
            },
            "pipeline_provenance_canon_sha256": "pipeline_digest_anchor",
            "model_provenance_canon_sha256": "model_digest_anchor",
        },
    )
    monkeypatch.setattr(shared_runner.infer_runtime, "run_sd3_inference", _fake_run_sd3_inference)

    prompt_run_root = tmp_path / "prompt_run_root"
    preview_result = shared_runner._prepare_source_pool_preview_artifact(
        cfg_obj={
            "policy_path": "content_np_geo_rescue",
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "model_source": "hf",
            "hf_revision": "main",
            "device": "cpu",
            "embed": {
                "preview_generation": {
                    "enabled": True,
                    "artifact_rel_path": "preview/preview.png",
                }
            },
        },
        prompt_run_root=prompt_run_root,
        prompt_text="parallel prompt 0",
        prompt_index=0,
        prompt_file_path="prompts/paper_small.txt",
    )
    preview_record = preview_result["preview_record"]
    preview_record_path = prompt_run_root / "artifacts" / "preview" / "preview_generation_record.json"
    persisted_preview_record = json.loads(preview_record_path.read_text(encoding="utf-8"))

    assert captured_inference_kwargs["runtime_phase_label"] == "preview_generation"
    assert preview_record["artifact_type"] == "stage_01_source_pool_preview_generation_record"
    assert preview_record["record_rel_path"] == "preview/preview_generation_record.json"
    assert preview_record["record_path"] == preview_record_path.as_posix()
    assert persisted_preview_record["artifact_type"] == "stage_01_source_pool_preview_generation_record"
    assert persisted_preview_record["record_rel_path"] == "preview/preview_generation_record.json"
    assert persisted_preview_record["inference_runtime_meta"]["cuda_memory_profile"]["phase_label"] == (
        "preview_generation"
    )
