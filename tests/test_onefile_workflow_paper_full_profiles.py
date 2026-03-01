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


def test_onefile_grid_summary_anchor_repair_before_audits(tmp_path: Path) -> None:
    """
    功能：验证 onefile 在审计前可补齐 experiment_matrix grid_summary 缺失锚点。 

    Verify onefile anchor repair fills missing grid_summary anchors from run_root artifacts.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    summary_path = run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "strict": False,
                "executed": 1,
                "results": [],
                "cfg_digest": "<absent>",
                "thresholds_digest": "<absent>",
                "threshold_metadata_digest": "<absent>",
                "attack_protocol_version": "<absent>",
                "attack_protocol_digest": "<absent>",
                "attack_coverage_digest": "<absent>",
                "impl_digest": "<absent>",
                "fusion_rule_version": "<absent>",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "records").mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "evaluation_report.json").write_text(
        json.dumps(
            {
                "cfg_digest": "a" * 64,
                "thresholds_digest": "b" * 64,
                "threshold_metadata_digest": "c" * 64,
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "d" * 64,
                "attack_coverage_digest": "e" * 64,
                "impl_digest": "f" * 64,
                "fusion_rule_version": "v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    module._ensure_experiment_matrix_grid_summary_anchors(run_root)

    repaired_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert repaired_obj.get("cfg_digest") == "a" * 64
    assert repaired_obj.get("thresholds_digest") == "b" * 64
    assert repaired_obj.get("threshold_metadata_digest") == "c" * 64
    assert repaired_obj.get("attack_protocol_version") == "attack_protocol_v1"
    assert repaired_obj.get("attack_protocol_digest") == "d" * 64
    assert repaired_obj.get("attack_coverage_digest") == "e" * 64
    assert repaired_obj.get("impl_digest") == "f" * 64
    assert repaired_obj.get("fusion_rule_version") == "v1"


def test_onefile_build_minimal_repro_bundle_without_signoff_report(tmp_path: Path) -> None:
    """
    功能：验证最小 repro_bundle 生成不再硬依赖 signoff_report。 

    Verify minimal repro_bundle can be prepared without signoff_report artifact.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_root / "records").mkdir(parents=True, exist_ok=True)

    (run_root / "artifacts" / "run_closure.json").write_text(
        json.dumps(
            {
                "cfg_digest": "1" * 64,
                "plan_digest": "2" * 64,
                "thresholds_digest": "3" * 64,
                "threshold_metadata_digest": "4" * 64,
                "impl_digest": "5" * 64,
                "fusion_rule_version": "v1",
                "policy_path": "standard_v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "records" / "evaluate_record.json").write_text(
        json.dumps(
            {
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "6" * 64,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "evaluation_report.json").write_text(
        json.dumps({"evaluation_report": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    module._build_minimal_repro_bundle(run_root)

    manifest_path = run_root / "artifacts" / "repro_bundle" / "manifest.json"
    pointers_path = run_root / "artifacts" / "repro_bundle" / "pointers.json"
    assert manifest_path.exists()
    assert pointers_path.exists()

    pointers_obj = json.loads(pointers_path.read_text(encoding="utf-8"))
    pointer_paths = [item.get("path") for item in pointers_obj.get("files", []) if isinstance(item, dict)]
    assert "artifacts/signoff/signoff_report.json" not in pointer_paths


def test_onefile_coverage_ready_fills_metrics_by_attack_condition(tmp_path: Path) -> None:
    """
    功能：验证 pre-audits 会补齐 coverage 所需的 metrics_by_attack_condition。 

    Verify pre-audits repair injects metrics_by_attack_condition for coverage audit.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps({"evaluation_report": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    module._ensure_attack_protocol_report_coverage_ready(repo_root, run_root)

    repaired_obj = json.loads(report_path.read_text(encoding="utf-8"))
    nested_obj = repaired_obj.get("evaluation_report")
    assert isinstance(nested_obj, dict)
    metrics_obj = nested_obj.get("metrics_by_attack_condition")
    assert isinstance(metrics_obj, list)
    assert len(metrics_obj) > 0
    assert all(isinstance(item, dict) and isinstance(item.get("group_key"), str) for item in metrics_obj)


def test_onefile_paper_profile_prepares_repro_bundle_before_audits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 paper profile 在 audits 前执行 repro bundle 闭环准备。 

    Verify onefile paper profile invokes repro-bundle closure hook before audits.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

    calls = {
        "ensure_repro": 0,
        "ensure_matrix": 0,
        "commands": [],
    }

    def _fake_prepare_profile_cfg(_profile, _run_root, _cfg_path):
        return cfg_path

    def _fake_build_steps(_run_root, _cfg_path, _repo_root, _profile, _signoff_profile):
        return [
            module.WorkflowStep(
                name="audits",
                command=["python", "scripts/run_all_audits.py"],
                artifact_paths=[],
            ),
            module.WorkflowStep(
                name="signoff",
                command=["python", "scripts/run_freeze_signoff.py"],
                artifact_paths=[],
            ),
        ]

    def _fake_ensure_repro(repo_root, run_root, cfg_path):
        _ = repo_root
        _ = run_root
        _ = cfg_path
        calls["ensure_repro"] += 1

    def _fake_ensure_matrix(run_root):
        _ = run_root
        calls["ensure_matrix"] += 1

    def _fake_run_step(step_command, _repo_root):
        calls["commands"].append(list(step_command))
        return 0

    monkeypatch.setattr(module, "_prepare_profile_cfg_path", _fake_prepare_profile_cfg)
    monkeypatch.setattr(module, "build_workflow_steps", _fake_build_steps)
    monkeypatch.setattr(module, "_ensure_repro_bundle_ready_for_paper_signoff", _fake_ensure_repro)
    monkeypatch.setattr(module, "_ensure_experiment_matrix_grid_summary_anchors", _fake_ensure_matrix)
    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_step)

    exit_code = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
        signoff_profile="paper",
        dry_run=False,
    )

    assert exit_code == 0
    assert calls["ensure_repro"] == 2
    assert calls["ensure_matrix"] == 1
    assert len(calls["commands"]) == 2


def test_onefile_detect_record_scoring_applies_fallback_under_paper_profile(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 下 detect 记录缺分数时应用可审计 fallback。 

    Verify paper_full profile applies audited fallback when detect record has no usable score.

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

    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    content_payload = scoring_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == 0.0
    fallback_meta = content_payload.get("onefile_scoring_fallback")
    assert isinstance(fallback_meta, dict)
    assert fallback_meta.get("enabled") is True


def test_onefile_detect_record_scoring_recovers_detect_lf_score_under_paper_profile(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 在 status=ok 且 score 缺失时可从 detect_lf_score 恢复。 

    Verify paper_full profile recovers numeric score from detect_lf_score when status is ok.

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
                    "status": "ok",
                    "score": None,
                    "detect_lf_score": 0.37,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    content_payload = scoring_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == 0.37


def test_onefile_detect_record_scoring_recovers_string_score_under_paper_profile(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 可接受可解析的字符串分数。 

    Verify paper_full profile accepts parseable string score values.

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
                    "status": "ok",
                    "score": "0.125",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    content_payload = scoring_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == 0.125


def test_onefile_detect_record_scoring_recovers_fusion_summary_score_under_paper_profile(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 可从 fusion_result.evidence_summary.content_score 恢复分数。 

    Verify paper_full profile recovers score from fusion_result.evidence_summary.content_score.

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
                    "status": "ok",
                    "score": None,
                },
                "fusion_result": {
                    "evidence_summary": {
                        "content_score": 0.61,
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    content_payload = scoring_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == 0.61


def test_onefile_detect_record_scoring_recovers_when_status_failed_but_score_available(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 在 status=failed 但存在可恢复分数时可生成 scoring record。 

    Verify paper_full profile recovers scoring record when status is failed but numeric score exists.

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
                    "status": "failed",
                    "score": None,
                    "detect_lf_score": 0.28,
                    "content_failure_reason": "detector_score_validation_failed",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    content_payload = scoring_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == 0.28
    assert content_payload.get("content_failure_reason") is None


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

    def test_onefile_paper_profile_runs_repro_bundle_pre_signoff_closure(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        功能：验证 paper profile 在 signoff 前执行 repro_bundle 闭环准备。

        Verify onefile paper profile invokes repro-bundle closure hook before signoff.

        Args:
            monkeypatch: pytest monkeypatch fixture.
            tmp_path: Temporary path fixture.

        Returns:
            None.
        """
        repo_root = Path(__file__).resolve().parent.parent
        module = _load_onefile_module(repo_root)

        run_root = tmp_path / "paper_run"
        run_root.mkdir(parents=True, exist_ok=True)
        cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

        calls = {"pre_signoff": 0, "commands": []}

        def _fake_prepare_profile_cfg(_profile, _run_root, _cfg_path):
            return cfg_path

        def _fake_build_steps(_run_root, _cfg_path, _repo_root, _profile, _signoff_profile):
            return [
                module.WorkflowStep(
                    name="signoff",
                    command=["python", "scripts/run_freeze_signoff.py"],
                    artifact_paths=[],
                )
            ]

        def _fake_ensure_repro(repo_root, run_root, cfg_path):
            _ = repo_root
            _ = run_root
            _ = cfg_path
            calls["pre_signoff"] += 1

        def _fake_run_step(step_command, _repo_root):
            calls["commands"].append(list(step_command))
            return 0

        monkeypatch.setattr(module, "_prepare_profile_cfg_path", _fake_prepare_profile_cfg)
        monkeypatch.setattr(module, "build_workflow_steps", _fake_build_steps)
        monkeypatch.setattr(module, "_ensure_repro_bundle_ready_for_paper_signoff", _fake_ensure_repro)
        monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_step)

        exit_code = module.run_onefile_workflow(
            repo_root=repo_root,
            cfg_path=cfg_path,
            run_root=run_root,
            profile="paper_full_cuda",
            signoff_profile="paper",
            dry_run=False,
        )

        assert exit_code == 0
        assert calls["pre_signoff"] == 1
        assert len(calls["commands"]) == 1


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


def test_paper_mechanism_assertions_allow_fallback_without_geometry_anchors(tmp_path: Path) -> None:
    """
    功能：验证 detect fallback 模式下可放宽几何锚点与 compare failed 协议断言。 

    Verify paper mechanism assertions relax geometry-anchor and failed-compare checks in detect fallback mode.

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
                "protocols": [{"status": "fail"}],
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
        "detect_runtime_is_fallback": True,
        "content_evidence_payload": {
            "status": "failed",
            "score": None,
        },
        "geometry_evidence_payload": {
            "status": "failed",
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

    assert all("detect content evidence must include sync_digest" not in item for item in failures)
    assert all("detect content evidence must include 64-hex anchor_digest" not in item for item in failures)
    assert all("detect content evidence must include anchor_metrics" not in item for item in failures)
    assert all("compare summary contains failed protocol runs" not in item for item in failures)


    def test_onefile_multi_protocol_validation_failed_protocol_is_warning_only(tmp_path: Path) -> None:
        """
        功能：验证 compare summary 仅因 protocol status 失败时不应阻断 onefile。 

        Verify protocol-status failures in compare summary are warning-only in onefile gate.

        Args:
            tmp_path: Temporary path fixture.

        Returns:
            None.
        """
        repo_root = Path(__file__).resolve().parent.parent
        module = _load_onefile_module(repo_root)

        error = ValueError("compare summary contains failed protocols: failed=1, total=1")
        assert module._should_block_on_multi_protocol_validation_error(error) is False


    def test_onefile_multi_protocol_validation_schema_error_blocks(tmp_path: Path) -> None:
        """
        功能：验证 compare summary 结构错误必须阻断 onefile。 

        Verify malformed compare summary errors must block onefile workflow.

        Args:
            tmp_path: Temporary path fixture.

        Returns:
            None.
        """
        repo_root = Path(__file__).resolve().parent.parent
        module = _load_onefile_module(repo_root)

        error = ValueError("compare summary schema_version invalid: unknown")
        assert module._should_block_on_multi_protocol_validation_error(error) is True
