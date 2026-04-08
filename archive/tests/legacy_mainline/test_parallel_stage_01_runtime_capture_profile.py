"""
文件目的：验证 parallel worker 路径会通过 shared baseline 子运行继承 runtime capture profile 与 run_closure command history。
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


PARALLEL_WORKER = _load_module(PARALLEL_WORKER_PATH, "test_parallel_stage_01_runtime_capture_profile_worker")


def _runtime_capture_profile(phase_label: str) -> Dict[str, Any]:
    """
    功能：构造最小 runtime capture 显存 profile。

    Build the minimal runtime capture CUDA memory profile used by the tests.

    Args:
        phase_label: Runtime phase label.

    Returns:
        CUDA memory profile mapping.
    """
    return {
        "status": "absent",
        "reason": "cuda_not_active",
        "phase_label": phase_label,
        "sample_scope": "single_worker_process_local",
        "device": "cpu",
    }


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


def test_parallel_worker_path_preserves_runtime_capture_profile_and_command_closures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：parallel worker 走 shared source-pool subrun 时，必须保留 embed_record 的 runtime capture profile 和 prompt-level run_closure 的 command_closures。

    Verify that the parallel worker path preserves the embed_record runtime
    capture profile and the prompt-level run_closure command history when it
    delegates to the shared source-pool subrun.

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

    config_path = tmp_path / "config.yaml"
    config_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")
    run_root = tmp_path / "run_root"

    monkeypatch.setattr(
        PARALLEL_WORKER,
        "load_yaml_mapping",
        lambda _path: {
            "policy_path": "content_np_geo_rescue",
            "embed": {
                "preview_generation": {
                    "enabled": True,
                    "artifact_rel_path": "preview/preview.png",
                }
            },
        },
    )
    monkeypatch.setattr(
        shared_runner,
        "_resolve_stage_01_source_pool_cfg",
        lambda _cfg: {"enabled": True, "record_usage": "stage_01_direct_source_pool"},
    )
    monkeypatch.setattr(
        shared_runner,
        "_resolve_stage_01_prompt_pool",
        lambda _cfg: (["parallel prompt 0"], "prompts/paper_small.txt"),
    )
    monkeypatch.setattr(shared_runner, "build_seed_audit", lambda *_args, **_kwargs: ({}, "seed_digest", 7, "seed_rule"))
    monkeypatch.setattr(
        shared_runner.pipeline_factory,
        "build_pipeline_shell",
        lambda _cfg: {
            "pipeline_obj": object(),
            "pipeline_status": "built",
            "pipeline_error": None,
            "pipeline_runtime_meta": {"model_source_resolution": "local_snapshot_priority"},
            "pipeline_provenance_canon_sha256": "pipeline_digest_anchor",
            "model_provenance_canon_sha256": "model_digest_anchor",
        },
    )
    monkeypatch.setattr(
        shared_runner.infer_runtime,
        "run_sd3_inference",
        lambda *_args, **kwargs: {
            "inference_status": shared_runner.infer_runtime.INFERENCE_STATUS_OK,
            "inference_error": None,
            "inference_runtime_meta": {
                "cuda_memory_profile": _runtime_capture_profile(str(kwargs.get("runtime_phase_label"))),
            },
            "output_image": _FakePreviewImage(),
        },
    )
    monkeypatch.setattr(shared_runner, "_build_stage_command", lambda stage_name, *_args: [stage_name])

    def _fake_run_stage(stage_name: str, _command: Any, prompt_run_root: Path) -> Dict[str, Any]:
        records_dir = prompt_run_root / "records"
        artifacts_dir = prompt_run_root / "artifacts"
        records_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if stage_name == "embed":
            _write_json(
                records_dir / "embed_record.json",
                {
                    "content_evidence": {
                        "audit": {
                            "runtime_capture_cuda_memory_profile": _runtime_capture_profile(
                                "statement_only_runtime_capture"
                            )
                        }
                    }
                },
            )
            _write_json(
                artifacts_dir / "run_closure.json",
                {
                    "command": "embed",
                    "status": {
                        "details": {
                            "runtime_finalization": {
                                "runtime_capture_cuda_memory_profile": _runtime_capture_profile(
                                    "statement_only_runtime_capture"
                                )
                            }
                        }
                    },
                    "command_closures": {
                        "embed": {
                            "command": "embed",
                            "status": {
                                "details": {
                                    "runtime_finalization": {
                                        "runtime_capture_cuda_memory_profile": _runtime_capture_profile(
                                            "statement_only_runtime_capture"
                                        )
                                    }
                                }
                            },
                        }
                    },
                },
            )
            return {"return_code": 0}

        _write_json(records_dir / "detect_record.json", {"payload_status": "ok"})
        _write_json(
            artifacts_dir / "run_closure.json",
            {
                "command": "detect",
                "status": {
                    "details": {
                        "runtime_finalization": {
                            "runtime_capture_cuda_memory_profile": _runtime_capture_profile(
                                "detect_main_inference"
                            )
                        }
                    }
                },
                "command_closures": {
                    "embed": {
                        "command": "embed",
                        "status": {
                            "details": {
                                "runtime_finalization": {
                                    "runtime_capture_cuda_memory_profile": _runtime_capture_profile(
                                        "statement_only_runtime_capture"
                                    )
                                }
                            }
                        },
                    },
                    "detect": {
                        "command": "detect",
                        "status": {
                            "details": {
                                "runtime_finalization": {
                                    "runtime_capture_cuda_memory_profile": _runtime_capture_profile(
                                        "detect_main_inference"
                                    )
                                }
                            }
                        },
                    },
                },
            },
        )
        return {"return_code": 0}

    monkeypatch.setattr(shared_runner, "_run_stage", _fake_run_stage)
    monkeypatch.setattr(
        shared_runner,
        "_normalize_direct_detect_payload",
        lambda payload, **kwargs: {
            **dict(payload),
            "label": True,
            "ground_truth": True,
            "is_watermarked": True,
            "prompt_index": kwargs["prompt_index"],
        },
    )
    monkeypatch.setattr(
        shared_runner,
        "_resolve_source_pool_attestation_views",
        lambda **_kwargs: {
            "attestation_statement": {"exists": False},
            "attestation_bundle": {"exists": False},
            "attestation_result": {"exists": False},
        },
    )
    monkeypatch.setattr(shared_runner, "_resolve_source_pool_source_image_view", lambda **_kwargs: {"exists": False})

    exit_code = PARALLEL_WORKER.run_parallel_worker(
        config_path,
        run_root,
        stage_run_id="parallel_stage_01",
        worker_index=0,
        worker_count=1,
    )

    prompt_run_root = run_root / "source_pool" / "prompt_000"
    preview_record = json.loads(
        (prompt_run_root / "artifacts" / "preview" / "preview_generation_record.json").read_text(encoding="utf-8")
    )
    embed_record = json.loads((prompt_run_root / "records" / "embed_record.json").read_text(encoding="utf-8"))
    staged_embed_record = json.loads(
        (
            run_root
            / shared_runner.SOURCE_POOL_EMBED_RECORDS_RELATIVE_ROOT
            / "000_embed_record.json"
        ).read_text(encoding="utf-8")
    )
    run_closure = json.loads((prompt_run_root / "artifacts" / "run_closure.json").read_text(encoding="utf-8"))
    worker_result = json.loads(
        (run_root / PARALLEL_WORKER.WORKER_RESULT_RELATIVE_ROOT / "worker_00_result.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert preview_record["artifact_type"] == "stage_01_source_pool_preview_generation_record"
    assert preview_record["record_rel_path"] == "preview/preview_generation_record.json"
    assert preview_record["inference_runtime_meta"]["cuda_memory_profile"]["phase_label"] == "preview_generation"
    assert embed_record["content_evidence"]["audit"]["runtime_capture_cuda_memory_profile"]["phase_label"] == (
        "statement_only_runtime_capture"
    )
    assert staged_embed_record["content_evidence"]["audit"]["runtime_capture_cuda_memory_profile"]["phase_label"] == (
        "statement_only_runtime_capture"
    )
    assert run_closure["command"] == "detect"
    assert run_closure["command_closures"]["embed"]["status"]["details"]["runtime_finalization"][
        "runtime_capture_cuda_memory_profile"
    ]["phase_label"] == "statement_only_runtime_capture"
    assert run_closure["command_closures"]["detect"]["status"]["details"]["runtime_finalization"][
        "runtime_capture_cuda_memory_profile"
    ]["phase_label"] == "detect_main_inference"
    assert worker_result["status"] == "ok"
    assert worker_result["completed_prompt_indices"] == [0]
