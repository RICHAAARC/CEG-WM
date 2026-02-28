"""
文件目的：onefile profile 隔离与 paper 机制断言回归测试。
Module type: General module
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_onefile_module(repo_root: Path):
    """
    功能：动态加载 onefile workflow 脚本模块。

    Dynamically load scripts/run_onefile_workflow.py as module.

    Args:
        repo_root: Repository root path.

    Returns:
        Imported module object.
    """
    module_path = repo_root / "scripts" / "run_onefile_workflow.py"
    spec = importlib.util.spec_from_file_location("run_onefile_workflow_profile_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_assert_script_module(repo_root: Path):
    """
    功能：动态加载 paper 机制断言脚本模块。

    Dynamically load scripts/assert_paper_mechanisms.py as module.

    Args:
        repo_root: Repository root path.

    Returns:
        Imported module object.
    """
    module_path = repo_root / "scripts" / "assert_paper_mechanisms.py"
    spec = importlib.util.spec_from_file_location("assert_paper_mechanisms_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_onefile_workflow_profiles_smoke_vs_paper_full_are_disjoint(tmp_path: Path) -> None:
    """
    功能：验证 smoke 与 paper_full profile 编排步骤严格隔离。

    Verify cpu_smoke and paper_full_cuda profiles are strictly disjoint in workflow steps.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    cfg_path = repo_root / "configs" / "default.yaml"

    smoke_steps = module.build_workflow_steps(
        run_root=run_root,
        cfg_path=cfg_path,
        repo_root=repo_root,
        profile="cpu_smoke",
        signoff_profile="baseline",
    )
    paper_steps = module.build_workflow_steps(
        run_root=run_root,
        cfg_path=cfg_path,
        repo_root=repo_root,
        profile="paper_full_cuda",
        signoff_profile="paper",
    )

    smoke_names = [step.name for step in smoke_steps]
    paper_names = [step.name for step in paper_steps]

    assert "assert_paper_mechanisms" not in smoke_names
    assert "multi_protocol_evaluation" not in smoke_names
    assert "assert_paper_mechanisms" in paper_names
    assert "multi_protocol_evaluation" in paper_names


def test_onefile_detect_record_scoring_fails_fast_under_paper_profile(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 下 detect 记录缺分数时必须 fail-fast。 

    Verify paper_full profile rejects detect records without valid score/status for calibration.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "absent",
                    "score": None,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paper_full_cuda requires detect_record"):
        module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")


def test_paper_full_mechanism_assertions_fail_fast_on_proxy_paths(tmp_path: Path) -> None:
    """
    功能：验证 paper 机制断言在 proxy 路径下 fail-fast。

    Verify mechanism assertion fails fast when proxy HF/LF configuration is used.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    assert_module = _load_assert_script_module(repo_root)

    run_root = tmp_path / "run_root"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare").mkdir(parents=True, exist_ok=True)

    cfg = {
        "paper_faithfulness": {"enabled": True},
        "impl": {
            "sync_module_id": "geometry_sync_baseline_v1",
            "geometry_extractor_id": "geometry_baseline_identity_v1",
        },
        "watermark": {
            "hf": {
                "enabled": True,
                "tail_truncation_mode": "gaussian",
                "selection": "winsor",
            },
            "lf": {
                "enabled": True,
                "coding_mode": "dct_proxy",
                "decoder": "majority_vote",
            },
        },
    }
    embed_record = {
        "content_evidence": {
            "trajectory_evidence": {"trajectory_spec_digest": "a" * 64},
            "injection_site_spec": {"status": "ok", "hook_type": "callback_on_step_end"},
            "lf_trace_digest": "b" * 64,
        }
    }
    detect_record = {"content_evidence_payload": {}}
    evaluate_report = {
        "attack_protocol_version": "attack_protocol_v1",
        "attack_protocol_digest": "c" * 64,
        "attack_coverage_digest": "d" * 64,
        "metrics_by_attack_condition": [{"group_key": "g0"}],
    }

    failures = assert_module._assert_paper_mechanisms(
        run_root=run_root,
        cfg=cfg,
        embed_record=embed_record,
        detect_record=detect_record,
        evaluate_report=evaluate_report,
        repo_root=repo_root,
    )

    assert any("tail_truncation_mode" in item for item in failures)
    assert any("coding_mode" in item for item in failures)
    assert any("decoder" in item for item in failures)


def test_paper_full_mechanism_assertions_accept_top_level_geometry_payload(tmp_path: Path) -> None:
    """
    功能：验证断言脚本可从 detect 顶层 geometry payload 读取几何锚点。 

    Verify mechanism assertion accepts geometry anchors from top-level geometry payload.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    assert_module = _load_assert_script_module(repo_root)

    run_root = tmp_path / "run_root"
    compare_dir = run_root / "artifacts" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "compare_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "protocol_compare_v1",
                "protocol_count": 1,
                "protocols": [{"status": "ok"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg = {
        "paper_faithfulness": {"enabled": True},
        "impl": {
            "sync_module_id": "geometry_latent_sync_sd3_v2",
            "geometry_extractor_id": "attention_anchor_map_relation_v1",
            "hf_embedder_id": "hf_embedder_t2smark_v1",
            "lf_coder_id": "lf_coder_prc_v1",
        },
        "watermark": {
            "hf": {
                "enabled": True,
                "tail_truncation_mode": "top_k_per_latent",
                "selection": "top_k_magnitude_based",
            },
            "lf": {
                "enabled": True,
                "coding_mode": "latent_space_sign_flipping",
                "decoder": "belief_propagation",
            },
        },
    }
    embed_record = {
        "content_evidence": {
            "injection_status": "ok",
            "injection_trace_digest": "1" * 64,
            "injection_digest": "2" * 64,
            "step_summary_digest": "3" * 64,
            "trajectory_evidence": {"trajectory_spec_digest": "4" * 64},
            "injection_site_spec": {"status": "ok", "hook_type": "callback_on_step_end"},
            "lf_trace_digest": "5" * 64,
            "hf_trace_digest": "6" * 64,
        }
    }
    detect_record = {
        "content_evidence_payload": {},
        "geometry_evidence_payload": {
            "sync_digest": "7" * 64,
            "anchor_digest": "8" * 64,
            "anchor_metrics": {"extraction_source": "attention_relation_summary"},
        },
    }
    evaluate_report = {
        "attack_protocol_version": "attack_protocol_v1",
        "attack_protocol_digest": "9" * 64,
        "attack_coverage_digest": "a" * 64,
        "metrics_by_attack_condition": [{"group_key": "g0"}],
    }

    failures = assert_module._assert_paper_mechanisms(
        run_root=run_root,
        cfg=cfg,
        embed_record=embed_record,
        detect_record=detect_record,
        evaluate_report=evaluate_report,
        repo_root=repo_root,
    )

    assert all("detect content evidence must include sync_digest" not in item for item in failures)
    assert all("detect content evidence must include anchor_digest" not in item for item in failures)
    assert all("geometry anchor_metrics must exist" not in item for item in failures)


def test_paper_full_mechanism_assertions_accept_nested_evaluation_report(tmp_path: Path) -> None:
    """
    功能：验证断言脚本兼容嵌套 evaluation_report 与可选 injection_site.status。 

    Verify mechanism assertion accepts nested evaluation_report payload and optional site status.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    assert_module = _load_assert_script_module(repo_root)

    run_root = tmp_path / "run_root"
    compare_dir = run_root / "artifacts" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "compare_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "protocol_compare_v1",
                "protocol_count": 1,
                "protocols": [{"status": "ok"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg = {
        "paper_faithfulness": {"enabled": True},
        "impl": {
            "sync_module_id": "geometry_latent_sync_sd3_v2",
            "geometry_extractor_id": "attention_anchor_map_relation_v1",
            "hf_embedder_id": "hf_embedder_t2smark_v1",
            "lf_coder_id": "lf_coder_prc_v1",
        },
        "watermark": {
            "hf": {
                "enabled": True,
                "tail_truncation_mode": "top_k_per_latent",
                "selection": "top_k_magnitude_based",
            },
            "lf": {
                "enabled": True,
                "coding_mode": "latent_space_sign_flipping",
                "decoder": "belief_propagation",
            },
        },
    }
    embed_record = {
        "content_evidence": {
            "injection_status": "ok",
            "injection_trace_digest": "1" * 64,
            "injection_digest": "2" * 64,
            "step_summary_digest": "3" * 64,
            "trajectory_evidence": {"trajectory_spec_digest": "4" * 64},
            "injection_site_spec": {"hook_type": "callback_on_step_end"},
        },
        "content_result": {
            "lf_trace_digest": "5" * 64,
            "hf_trace_digest": "6" * 64,
        },
    }
    detect_record = {
        "content_evidence_payload": {},
        "geometry_evidence_payload": {
            "sync_digest": "7" * 64,
            "anchor_digest": "8" * 64,
            "anchor_metrics": {"extraction_source": "attention_relation_summary"},
        },
    }
    evaluate_report = {
        "evaluation_report": {
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "9" * 64,
            "attack_coverage_digest": "a" * 64,
            "metrics_by_attack_condition": [{"group_key": "g0"}],
        }
    }

    failures = assert_module._assert_paper_mechanisms(
        run_root=run_root,
        cfg=cfg,
        embed_record=embed_record,
        detect_record=detect_record,
        evaluate_report=evaluate_report,
        repo_root=repo_root,
    )

    assert all("injection_site_spec.status must indicate bound/ok" not in item for item in failures)
    assert all("content_evidence.lf_trace_digest must exist" not in item for item in failures)
    assert all("content_evidence.hf_trace_digest must exist when hf enabled" not in item for item in failures)
    assert all("evaluation_report.attack_protocol_version must exist" not in item for item in failures)
    assert all("evaluation_report.attack_protocol_digest must exist" not in item for item in failures)
    assert all("evaluation_report.attack_coverage_digest must exist" not in item for item in failures)
    assert all("evaluation_report.metrics_by_attack_condition must be non-empty list" not in item for item in failures)


def test_paper_full_mechanism_assertions_reject_latent_mode_sync_only_geometry(tmp_path: Path) -> None:
    """
    功能：验证 latent per-step 模式下缺失 anchor 证据会触发硬失败。 

    Verify latent per-step mode rejects sync-only geometry evidence when anchor evidence is absent.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    assert_module = _load_assert_script_module(repo_root)

    run_root = tmp_path / "run_root"
    compare_dir = run_root / "artifacts" / "multi_protocol_evaluation" / "artifacts" / "protocol_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "compare_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "protocol_compare_v1",
                "protocol_count": 1,
                "protocols": [{"status": "ok"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg = {
        "paper_faithfulness": {"enabled": True},
        "impl": {
            "sync_module_id": "geometry_latent_sync_sd3_v2",
            "geometry_extractor_id": "attention_anchor_map_relation_v1",
            "hf_embedder_id": "hf_embedder_t2smark_v1",
            "lf_coder_id": "lf_coder_prc_v1",
        },
        "watermark": {
            "hf": {
                "enabled": True,
                "tail_truncation_mode": "top_k_per_latent",
                "selection": "top_k_magnitude_based",
            },
            "lf": {
                "enabled": True,
                "coding_mode": "latent_space_sign_flipping",
                "decoder": "belief_propagation",
            },
        },
    }
    embed_record = {
        "content_evidence": {
            "injection_status": "ok",
            "injection_trace_digest": "1" * 64,
            "injection_digest": "2" * 64,
            "step_summary_digest": "3" * 64,
            "trajectory_evidence": {"trajectory_spec_digest": "4" * 64},
            "injection_site_spec": {"hook_type": "callback_on_step_end"},
        },
        "embed_trace": {
            "embed_mode": "latent_step_injection_v1",
        },
    }
    detect_record = {
        "content_evidence_payload": {},
        "geometry_evidence_payload": {
            "sync_digest": "7" * 64,
        },
    }
    evaluate_report = {
        "evaluation_report": {
            "attack_protocol_version": "attack_protocol_v1",
            "attack_protocol_digest": "9" * 64,
            "attack_coverage_digest": "a" * 64,
            "metrics_by_attack_condition": [{"group_key": "g0"}],
        }
    }

    failures = assert_module._assert_paper_mechanisms(
        run_root=run_root,
        cfg=cfg,
        embed_record=embed_record,
        detect_record=detect_record,
        evaluate_report=evaluate_report,
        repo_root=repo_root,
    )

    assert all("content_evidence.lf_trace_digest must exist" not in item for item in failures)
    assert all("content_evidence.hf_trace_digest must exist when hf enabled" not in item for item in failures)
    assert any("detect content evidence must include 64-hex anchor_digest" in item for item in failures)
    assert any("detect content evidence must include anchor_metrics" in item for item in failures)


def test_assert_multi_protocol_compare_success_rejects_failed_protocol(tmp_path: Path) -> None:
    """
    功能：验证 compare summary 存在失败协议时 paper 机制断言必须失败。

    Verify compare summary with failed protocol status is rejected by paper assertion helper.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    assert_module = _load_assert_script_module(repo_root)

    compare_summary_path = tmp_path / "compare_summary.json"
    compare_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "protocol_compare_v1",
                "protocols": [{"status": "ok"}, {"status": "fail"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = assert_module._assert_multi_protocol_compare_success(compare_summary_path)
    assert any("failed protocol runs" in item for item in failures)
