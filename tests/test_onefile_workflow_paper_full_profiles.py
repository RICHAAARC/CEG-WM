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
import yaml


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

    module._ensure_attack_protocol_report_coverage_ready(repo_root, run_root, "cpu_smoke")

    repaired_obj = json.loads(report_path.read_text(encoding="utf-8"))
    nested_obj = repaired_obj.get("evaluation_report")
    assert isinstance(nested_obj, dict)
    metrics_obj = nested_obj.get("metrics_by_attack_condition")
    assert isinstance(metrics_obj, list)
    assert len(metrics_obj) > 0
    assert all(isinstance(item, dict) and isinstance(item.get("group_key"), str) for item in metrics_obj)


def test_onefile_coverage_ready_replaces_unknown_only_metrics(tmp_path: Path) -> None:
    """
    功能：验证仅有 unknown sentinel 条目时，pre-audits 仍会补齐声明条件。 

    Verify pre-audits fills declared conditions when report only has unknown sentinel.

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
        json.dumps(
            {
                "evaluation_report": {
                    "metrics_by_attack_condition": [
                        {
                            "group_key": "unknown_attack::unknown_params"
                        }
                    ]
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    module._ensure_attack_protocol_report_coverage_ready(repo_root, run_root, "cpu_smoke")

    repaired_obj = json.loads(report_path.read_text(encoding="utf-8"))
    nested_obj = repaired_obj.get("evaluation_report")
    assert isinstance(nested_obj, dict)
    metrics_obj = nested_obj.get("metrics_by_attack_condition")
    assert isinstance(metrics_obj, list)
    group_keys = {
        item.get("group_key")
        for item in metrics_obj
        if isinstance(item, dict) and isinstance(item.get("group_key"), str)
    }
    assert "unknown_attack::unknown_params" not in group_keys
    assert len(group_keys) > 0


def test_prepare_profile_cfg_path_sets_paper_matrix_baseline_only(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda 运行期配置将 experiment_matrix 固定为 baseline-only。 

    Verify _prepare_profile_cfg_path forces experiment_matrix.ablation_variants to [{}]
    in paper_full_cuda runtime profile config.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

    profile_cfg_path = module._prepare_profile_cfg_path(
        "paper_full_cuda",
        run_root,
        cfg_path,
    )

    profile_cfg_obj = module.yaml.safe_load(profile_cfg_path.read_text(encoding="utf-8"))
    assert isinstance(profile_cfg_obj, dict)

    matrix_cfg = profile_cfg_obj.get("experiment_matrix")
    assert isinstance(matrix_cfg, dict)
    assert matrix_cfg.get("ablation_variants") == [{}]


def test_prepare_experiment_matrix_cfg_path_disables_paper_faithfulness(tmp_path: Path) -> None:
    """
    功能：验证 experiment_matrix 专用配置关闭 paper faithfulness。

    Verify matrix-specific config disables paper_faithfulness for paper_full_cuda.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

    profile_cfg_path = module._prepare_profile_cfg_path("paper_full_cuda", run_root, cfg_path)
    matrix_cfg_path = module._prepare_experiment_matrix_cfg_path("paper_full_cuda", run_root, profile_cfg_path)
    matrix_cfg_obj = module.yaml.safe_load(matrix_cfg_path.read_text(encoding="utf-8"))

    assert isinstance(matrix_cfg_obj, dict)
    paper_cfg = matrix_cfg_obj.get("paper_faithfulness")
    assert isinstance(paper_cfg, dict)
    assert paper_cfg.get("enabled") is False
    assert paper_cfg.get("alignment_check") is False


def test_run_onefile_workflow_uses_matrix_specific_cfg_for_experiment_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 onefile 在 experiment_matrix 步骤使用 matrix 专用配置。

    Verify run_onefile_workflow replaces experiment_matrix --config with
    matrix-specific config path under paper_full_cuda profile.

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

    effective_cfg_path = run_root / "artifacts" / "workflow_cfg" / "profile_paper_full_cuda.yaml"
    matrix_cfg_path = run_root / "artifacts" / "workflow_cfg" / "experiment_matrix_config.yaml"
    effective_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    effective_cfg_path.write_text("paper_faithfulness:\n  enabled: true\n", encoding="utf-8")
    matrix_cfg_path.write_text("paper_faithfulness:\n  enabled: false\n", encoding="utf-8")

    captured_commands: list[list[str]] = []

    def _fake_prepare_profile_cfg(_profile, _run_root, _cfg_path):
        return effective_cfg_path

    def _fake_prepare_matrix_cfg(_profile, _run_root, _cfg_path):
        return matrix_cfg_path

    def _fake_build_steps(_run_root, _cfg_path, _repo_root, _profile, _signoff_profile):
        return [
            module.WorkflowStep(
                name="experiment_matrix",
                command=[
                    "python",
                    str(repo_root / "scripts" / "run_experiment_matrix.py"),
                    "--config",
                    str(effective_cfg_path),
                    "--batch-root",
                    str(run_root / "outputs" / "experiment_matrix"),
                ],
                artifact_paths=[run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json"],
            )
        ]

    def _fake_run_step(step_command, _repo_root):
        captured_commands.append(list(step_command))
        return 0

    def _fake_load_summary(_summary_path: Path) -> dict:
        return {"total": 1, "failed": 0, "succeeded": 1}

    monkeypatch.setattr(module, "_prepare_profile_cfg_path", _fake_prepare_profile_cfg)
    monkeypatch.setattr(module, "_prepare_experiment_matrix_cfg_path", _fake_prepare_matrix_cfg)
    monkeypatch.setattr(module, "build_workflow_steps", _fake_build_steps)
    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_step)
    monkeypatch.setattr(module, "_load_experiment_matrix_summary", _fake_load_summary)

    exit_code = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
        signoff_profile="paper",
        dry_run=False,
    )

    assert exit_code == 0
    assert captured_commands, "experiment_matrix step command must be executed"
    command = captured_commands[0]
    assert "--config" in command
    config_idx = command.index("--config")
    assert command[config_idx + 1] == str(matrix_cfg_path)


def test_onefile_pre_audits_order_runs_repro_after_coverage_and_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 audits 前预修复顺序为 coverage/matrix 在前，repro_bundle 在后。 

    Verify pre-audits order runs repro bundle after coverage and matrix repairs.

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

    call_order = []

    def _fake_prepare_profile_cfg(_profile, _run_root, _cfg_path):
        return cfg_path

    def _fake_build_steps(_run_root, _cfg_path, _repo_root, _profile, _signoff_profile):
        return [
            module.WorkflowStep(
                name="audits",
                command=["python", "scripts/run_all_audits.py"],
                artifact_paths=[],
            ),
        ]

    def _fake_coverage(_repo_root, _run_root, _profile):
        call_order.append("coverage")

    def _fake_matrix(_run_root):
        call_order.append("matrix")

    def _fake_repro(repo_root, run_root, cfg_path):
        _ = repo_root
        _ = run_root
        _ = cfg_path
        call_order.append("repro")

    def _fake_run_step(_step_command, _repo_root):
        return 0

    monkeypatch.setattr(module, "_prepare_profile_cfg_path", _fake_prepare_profile_cfg)
    monkeypatch.setattr(module, "build_workflow_steps", _fake_build_steps)
    monkeypatch.setattr(module, "_ensure_attack_protocol_report_coverage_ready", _fake_coverage)
    monkeypatch.setattr(module, "_ensure_experiment_matrix_grid_summary_anchors", _fake_matrix)
    monkeypatch.setattr(module, "_ensure_repro_bundle_ready_for_paper_signoff", _fake_repro)
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
    assert call_order == ["coverage", "matrix", "repro"]


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
    功能：验证 paper_full profile 下 detect 记录无任何可恢复分数时拒绝 fallback（抛出 ValueError）。 

    Verify paper_full profile raises ValueError when detect record has no usable score,
    since paper mode forbids fallback to prevent polluting NP calibration distribution.

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

    # paper 模式禁止无分数时的 fallback，拒绝继续以防污染 NP 校准输入分布。
    import pytest as _pytest
    with _pytest.raises(ValueError, match="content_evidence_payload"):
        module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")


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
    功能：验证 paper_full profile 在 status=failed 时拒绝 score 恢复，保留失败语义。

    Verify paper_full profile raises ValueError when status is failed,
    even if a numeric score field exists. Failed evidence must not be
    rewritten as calibratable score.

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

    # paper 模式下 status=failed 证据不得被改写为 ok，必须抛出 ValueError。
    import pytest as _pytest
    with _pytest.raises(ValueError, match="content_evidence_payload"):
        module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")


def test_onefile_detect_record_scoring_allows_hf_trace_recovery_with_failed_semantics(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 在 mask_extraction_no_input 下允许使用 HF trace 分数恢复，但不改写失败语义。

    Verify paper_full profile may recover score from hf_detect_trace.hf_score_raw
    when content failure reason is mask_extraction_no_input and hf_status is ok,
    while preserving failed status semantics.

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
                    "content_failure_reason": "mask_extraction_no_input",
                    "score_parts": {
                        "hf_status": "ok",
                        "lf_status": "absent",
                        "hf_detect_trace": {
                            "hf_status": "ok",
                            "hf_score_raw": -0.0041727437,
                        },
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    assert scoring_path.exists()
    assert scoring_path.name == "detect_record_for_scoring.json"

    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    content_payload = scoring_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "failed"
    assert content_payload.get("content_failure_reason") == "mask_extraction_no_input"
    assert content_payload.get("score") == -0.0041727437
    assert content_payload.get("calibration_score_recovery_source") == "score_parts.hf_detect_trace.hf_score_raw"


def test_onefile_detect_record_scoring_allows_fallback_when_sidecar_disabled(tmp_path: Path) -> None:
    """
    功能：验证 paper_full profile 在 sidecar 禁用时仅写诊断补全，不污染正式校准输入。

    Verify paper_full profile writes diagnostic fallback sample when both LF/HF are
    absent due to image_domain_sidecar_disabled, while returning original detect record
    for formal calibration input.

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
                    "detect_lf_score": None,
                    "lf_score": None,
                    "content_failure_reason": "mask_extraction_no_input",
                    "score_parts": {
                        "lf_status": "absent",
                        "hf_status": "absent",
                        "lf_detect_trace": {
                            "lf_status": "absent",
                            "lf_absent_reason": "image_domain_sidecar_disabled",
                        },
                        "hf_detect_trace": {
                            "hf_status": "absent",
                            "hf_absent_reason": "image_domain_sidecar_disabled",
                        },
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # sidecar 禁用是配置性缺失：写诊断补全文件，并生成带来源标记的正式校准输入。
    scoring_path = module._prepare_detect_record_for_scoring(run_root, records_dir, "paper_full_cuda")
    assert scoring_path.exists()
    assert scoring_path != detect_record_path
    assert scoring_path.name == "detect_record_for_calibration.json"

    scoring_payload = json.loads(scoring_path.read_text(encoding="utf-8"))
    scoring_content_payload = scoring_payload.get("content_evidence_payload", {})
    assert scoring_content_payload.get("score") == 0.0
    assert scoring_content_payload.get("status") == "ok"
    assert scoring_content_payload.get("calibration_sample_origin") == "sidecar_disabled_fallback"
    assert "calibration_sample_is_synthetic_fallback" not in scoring_content_payload
    assert scoring_content_payload.get("calibration_sample_usage") == "formal_with_sidecar_disabled_marker"

    diagnostic_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_calibration_diagnostic.json"
    assert diagnostic_path.exists()
    diagnostic_payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    content_payload = diagnostic_payload.get("content_evidence_payload", {})
    assert content_payload.get("score") == 0.0
    assert content_payload.get("status") == "ok"
    assert content_payload.get("calibration_sample_origin") == "sidecar_disabled_fallback"
    assert content_payload.get("calibration_sample_is_synthetic_fallback") is True


def test_onefile_prepare_stage_cfg_path_injects_minimal_ground_truth_bundle(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda 的 calibrate/evaluate 阶段会注入最小 GT bundle。 

    Verify paper_full_cuda stage config injects minimal labeled detect records glob.

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
                    "score": 0.5,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(
        "calibration:\n  detect_records_glob: null\n"
        "evaluate:\n  detect_records_glob: null\n  thresholds_path: null\n",
        encoding="utf-8",
    )

    calibrate_cfg_path = module._prepare_stage_cfg_path("calibrate", run_root, cfg_path, "paper_full_cuda")
    calibrate_cfg_obj = module.yaml.safe_load(calibrate_cfg_path.read_text(encoding="utf-8"))
    calibrate_glob = calibrate_cfg_obj.get("calibration", {}).get("detect_records_glob")
    assert isinstance(calibrate_glob, str)
    assert "detect_records_calibrate_gt_*.json" in calibrate_glob

    positive_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_positive.json"
    negative_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_negative.json"
    assert positive_path.exists()
    assert negative_path.exists()

    positive_payload = json.loads(positive_path.read_text(encoding="utf-8"))
    negative_payload = json.loads(negative_path.read_text(encoding="utf-8"))
    assert positive_payload.get("is_watermarked") is True
    assert negative_payload.get("is_watermarked") is False

    evaluate_cfg_path = module._prepare_stage_cfg_path("evaluate", run_root, cfg_path, "paper_full_cuda")
    evaluate_cfg_obj = module.yaml.safe_load(evaluate_cfg_path.read_text(encoding="utf-8"))
    evaluate_glob = evaluate_cfg_obj.get("evaluate", {}).get("detect_records_glob")
    assert isinstance(evaluate_glob, str)
    assert "detect_records_evaluate_gt_*.json" in evaluate_glob


def test_onefile_prepare_stage_cfg_path_supports_multiple_ground_truth_pairs(tmp_path: Path) -> None:
    """
    功能：验证可通过配置生成多对 GT records bundle。 

    Verify stage config can generate multiple GT record pairs via pair_count.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root_multi"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.55,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "paper_cfg_multi.yaml"
    cfg_path.write_text(
        "calibration:\n"
        "  detect_records_glob: null\n"
        "  minimal_ground_truth_pair_count: 3\n"
        "evaluate:\n"
        "  detect_records_glob: null\n"
        "  thresholds_path: null\n"
        "  minimal_ground_truth_pair_count: 2\n",
        encoding="utf-8",
    )

    calibrate_cfg_path = module._prepare_stage_cfg_path("calibrate", run_root, cfg_path, "paper_full_cuda")
    calibrate_cfg_obj = module.yaml.safe_load(calibrate_cfg_path.read_text(encoding="utf-8"))
    calibrate_glob = calibrate_cfg_obj.get("calibration", {}).get("detect_records_glob")
    assert isinstance(calibrate_glob, str)
    assert "detect_records_calibrate_gt_*.json" in calibrate_glob

    workflow_cfg_dir = run_root / "artifacts" / "workflow_cfg"
    calibrate_files = sorted(workflow_cfg_dir.glob("detect_records_calibrate_gt_*.json"))
    assert len(calibrate_files) == 6

    evaluate_cfg_path = module._prepare_stage_cfg_path("evaluate", run_root, cfg_path, "paper_full_cuda")
    evaluate_cfg_obj = module.yaml.safe_load(evaluate_cfg_path.read_text(encoding="utf-8"))
    evaluate_glob = evaluate_cfg_obj.get("evaluate", {}).get("detect_records_glob")
    assert isinstance(evaluate_glob, str)
    assert "detect_records_evaluate_gt_*.json" in evaluate_glob

    evaluate_files = sorted(workflow_cfg_dir.glob("detect_records_evaluate_gt_*.json"))
    assert len(evaluate_files) == 4


def test_onefile_prepare_stage_cfg_path_propagates_dual_branch_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 dual-branch 失败后回退 clone 的 GT 记录会写入失败原因审计字段。

    Verify stage config fallback writes dual_branch_failure_reason into clone GT records.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root_failure_reason"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.42,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "paper_cfg_failure_reason.yaml"
    cfg_path.write_text(
        "calibration:\n"
        "  detect_records_glob: null\n"
        "evaluate:\n"
        "  detect_records_glob: null\n"
        "  thresholds_path: null\n",
        encoding="utf-8",
    )

    def _fake_dual_branch_failure(*args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        raise RuntimeError("dual_branch detect failed with return code 1")

    monkeypatch.setattr(module, "_run_dual_branch_embedding_and_detection", _fake_dual_branch_failure)

    calibrate_cfg_path = module._prepare_stage_cfg_path(
        "calibrate",
        run_root,
        cfg_path,
        "paper_full_cuda",
        repo_root,
    )
    assert calibrate_cfg_path.exists()

    positive_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_positive.json"
    negative_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_negative.json"
    assert positive_path.exists()
    assert negative_path.exists()

    positive_payload = json.loads(positive_path.read_text(encoding="utf-8"))
    negative_payload = json.loads(negative_path.read_text(encoding="utf-8"))
    assert "dual_branch_failure_reason" in positive_payload
    assert "dual_branch_failure_reason" in negative_payload
    assert "RuntimeError" in str(positive_payload.get("dual_branch_failure_reason"))


def test_dual_branch_embed_uses_whitelisted_test_mode_identity_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 dual-branch 负样本 embed 使用 whitelist 允许的 test_mode_identity 覆写。 

    Verify dual-branch negative embed uses whitelisted test_mode_identity override
    instead of forbidden embed.injection_enabled override.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

    calls = []

    def _fake_run_step(command, _repo_root):
        calls.append(list(command))
        command_text = " ".join(str(item) for item in command)
        branch_neg_root = run_root / "artifacts" / "branch_neg"
        records_dir = branch_neg_root / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        if "-m main.cli.run_embed" in command_text:
            (records_dir / "embed_record.json").write_text("{}", encoding="utf-8")
            return 0
        if "-m main.cli.run_detect" in command_text:
            (records_dir / "detect_record.json").write_text("{}", encoding="utf-8")
            return 0
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_step)

    _, detect_path = module._run_dual_branch_embedding_and_detection(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
    )

    assert detect_path.exists()
    embed_commands = [command for command in calls if "-m" in command and "main.cli.run_embed" in command]
    assert embed_commands
    embed_cfg_index = embed_commands[0].index("--config") + 1
    embed_cfg_path = Path(str(embed_commands[0][embed_cfg_index]))
    embed_cfg_obj = yaml.safe_load(embed_cfg_path.read_text(encoding="utf-8"))
    assert isinstance(embed_cfg_obj, dict)
    paper_cfg = embed_cfg_obj.get("paper_faithfulness")
    assert isinstance(paper_cfg, dict)
    assert paper_cfg.get("enabled") is False
    assert paper_cfg.get("alignment_check") is False

    assert "test_mode_identity=true" in embed_commands[0]
    assert "enable_paper_faithfulness=false" not in embed_commands[0]
    assert "enable_paper_faithfulness=true" not in embed_commands[0]
    assert "embed.injection_enabled=false" not in embed_commands[0]

    detect_commands = [command for command in calls if "-m" in command and "main.cli.run_detect" in command]
    assert detect_commands
    assert detect_commands[0].count("--input") == 1


def test_prepare_detect_records_recovers_dual_branch_negative_score_for_calibration(tmp_path: Path) -> None:
    """
    功能：验证 dual-branch 负样本记录在 score 缺失时可从 hf_score_raw 恢复为校准可用样本。 

    Verify minimal GT preparation recovers negative score from hf_detect_trace.hf_score_raw
    and marks it as formal dual-branch recovery sample.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    run_root.mkdir(parents=True, exist_ok=True)

    source_detect_path = run_root / "records" / "detect_record_for_scoring.json"
    source_detect_path.parent.mkdir(parents=True, exist_ok=True)
    source_detect_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "failed",
                    "score": -0.1,
                    "content_failure_reason": "mask_extraction_no_input",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    branch_neg_detect_path = run_root / "artifacts" / "branch_neg" / "records" / "detect_record.json"
    branch_neg_detect_path.parent.mkdir(parents=True, exist_ok=True)
    branch_neg_detect_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "failed",
                    "score": None,
                    "score_parts": {
                        "hf_detect_trace": {
                            "hf_status": "ok",
                            "hf_score_raw": -0.023,
                        }
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    glob_pattern = module._prepare_detect_records_with_minimal_ground_truth(
        run_root=run_root,
        source_detect_path=source_detect_path,
        stage_name="calibrate",
        branch_neg_detect_record=branch_neg_detect_path,
    )
    assert "detect_records_calibrate_gt_*.json" in glob_pattern

    negative_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_negative.json"
    assert negative_path.exists()
    negative_payload = json.loads(negative_path.read_text(encoding="utf-8"))
    content_payload = negative_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == -0.023
    assert content_payload.get("calibration_sample_origin") == "dual_branch_negative_recovery"
    assert content_payload.get("calibration_sample_usage") == "formal_with_dual_branch_negative_marker"


def test_prepare_detect_records_clone_mode_recovers_failed_source_for_negative_gt(tmp_path: Path) -> None:
    """
    功能：验证单分支 clone 回退场景可从 failed 源恢复负样本 GT 分数。 

    Verify clone-based GT generation can recover negative sample when source
    detect record is failed but already has recovered score metadata.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root_clone_recovery"
    source_detect_path = run_root / "records" / "detect_record_for_scoring.json"
    source_detect_path.parent.mkdir(parents=True, exist_ok=True)
    source_detect_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "failed",
                    "score": -0.0042,
                    "content_failure_reason": "mask_extraction_no_input",
                    "calibration_score_recovery_reason": "mask_extraction_no_input_with_hf_trace",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    glob_pattern = module._prepare_detect_records_with_minimal_ground_truth(
        run_root=run_root,
        source_detect_path=source_detect_path,
        stage_name="calibrate",
    )
    assert "detect_records_calibrate_gt_*.json" in glob_pattern

    negative_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_negative.json"
    assert negative_path.exists()
    negative_payload = json.loads(negative_path.read_text(encoding="utf-8"))
    content_payload = negative_payload.get("content_evidence_payload", {})
    assert content_payload.get("status") == "ok"
    assert content_payload.get("score") == -1.0
    assert content_payload.get("calibration_sample_usage") == "synthetic_negative_for_ground_truth_closure"
    assert content_payload.get("calibration_sample_origin") == "synthetic_negative_bundle_from_failed_source_v1"


def test_prepare_detect_records_clone_mode_normalizes_positive_from_recovered_failed_source(tmp_path: Path) -> None:
    """
    功能：验证单分支 clone 回退场景会将 recovered failed 源正样本标准化为 status=ok。 

    Verify clone-based GT generation normalizes positive sample to status=ok
    when source detect record is failed but already has recovered score metadata.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root_clone_positive_recovery"
    source_detect_path = run_root / "records" / "detect_record_for_scoring.json"
    source_detect_path.parent.mkdir(parents=True, exist_ok=True)
    source_detect_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "failed",
                    "score": -0.0042,
                    "content_failure_reason": "mask_extraction_no_input",
                    "calibration_score_recovery_reason": "mask_extraction_no_input_with_hf_trace",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    module._prepare_detect_records_with_minimal_ground_truth(
        run_root=run_root,
        source_detect_path=source_detect_path,
        stage_name="evaluate",
    )

    positive_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_evaluate_gt_positive.json"
    assert positive_path.exists()
    positive_payload = json.loads(positive_path.read_text(encoding="utf-8"))
    positive_content = positive_payload.get("content_evidence_payload", {})
    assert positive_content.get("status") == "ok"
    assert positive_content.get("content_failure_reason") is None
    assert positive_content.get("score") == -0.0042
    assert positive_content.get("calibration_sample_origin") == "formal_positive_recovered_from_failed_source_v1"


def test_prepare_detect_records_clone_mode_keeps_dual_branch_failure_reason(tmp_path: Path) -> None:
    """
    功能：验证 clone 回退时会把 dual-branch 失败原因写入 GT 审计字段。

    Verify clone fallback records include dual_branch_failure_reason audit field.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root_clone_reason"
    source_detect_path = run_root / "records" / "detect_record_for_scoring.json"
    source_detect_path.parent.mkdir(parents=True, exist_ok=True)
    source_detect_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.5,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failure_reason = "RuntimeError: dual_branch detect failed with return code 1"
    module._prepare_detect_records_with_minimal_ground_truth(
        run_root=run_root,
        source_detect_path=source_detect_path,
        stage_name="calibrate",
        dual_branch_failure_reason=failure_reason,
    )

    positive_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_positive.json"
    negative_path = run_root / "artifacts" / "workflow_cfg" / "detect_records_calibrate_gt_negative.json"
    positive_payload = json.loads(positive_path.read_text(encoding="utf-8"))
    negative_payload = json.loads(negative_path.read_text(encoding="utf-8"))
    assert positive_payload.get("dual_branch_failure_reason") == failure_reason
    assert negative_payload.get("dual_branch_failure_reason") == failure_reason


def test_onefile_prepare_stage_cfg_path_uses_prompt_list_as_gt_driver(tmp_path: Path) -> None:
    """
    功能：验证可通过 prompts 文件驱动 GT records 生成与 prompt 注入。

    Verify prompts file drives GT pair count and prompt injection into generated records.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root_prompt_file"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    detect_record_path = records_dir / "detect_record.json"
    detect_record_path.write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.61,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    prompts_path = tmp_path / "prompts_3.txt"
    prompts_path.write_text("prompt one\nprompt two\nprompt three\n", encoding="utf-8")

    cfg_path = tmp_path / "paper_cfg_prompt_file.yaml"
    cfg_path.write_text(
        "calibration:\n"
        "  detect_records_glob: null\n"
        "  minimal_ground_truth_pair_count: 99\n"
        f"  minimal_ground_truth_prompts_file: \"{prompts_path.name}\"\n"
        "evaluate:\n"
        "  detect_records_glob: null\n"
        "  thresholds_path: null\n",
        encoding="utf-8",
    )

    calibrate_cfg_path = module._prepare_stage_cfg_path("calibrate", run_root, cfg_path, "paper_full_cuda")
    calibrate_cfg_obj = module.yaml.safe_load(calibrate_cfg_path.read_text(encoding="utf-8"))
    calibrate_glob = calibrate_cfg_obj.get("calibration", {}).get("detect_records_glob")
    assert isinstance(calibrate_glob, str)
    assert "detect_records_calibrate_gt_*.json" in calibrate_glob

    workflow_cfg_dir = run_root / "artifacts" / "workflow_cfg"
    calibrate_files = sorted(workflow_cfg_dir.glob("detect_records_calibrate_gt_*.json"))
    assert len(calibrate_files) == 6

    positive_file = next(item for item in calibrate_files if "positive_000" in item.name)
    negative_file = next(item for item in calibrate_files if "negative_000" in item.name)
    positive_payload = json.loads(positive_file.read_text(encoding="utf-8"))
    negative_payload = json.loads(negative_file.read_text(encoding="utf-8"))
    assert positive_payload.get("inference_prompt") == "prompt one"
    assert negative_payload.get("inference_prompt") == "prompt one"


def test_onefile_resolve_prompt_file_path_fallbacks_to_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 prompt 相对路径在 cfg 目录缺失时可回退到仓库根目录解析。

    Verify prompt path resolution falls back to repo root when cfg-relative file is absent.

    Args:
        tmp_path: Temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    fake_repo_root = tmp_path / "fake_repo"
    cfg_dir = fake_repo_root / "configs"
    prompts_dir = fake_repo_root / "prompts"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_file = prompts_dir / "paper_small.txt"
    prompt_file.write_text("prompt-a\n", encoding="utf-8")

    cfg_path = cfg_dir / "paper_full_cuda.yaml"
    cfg_path.write_text("inference_prompt_file: \"prompts/paper_small.txt\"\n", encoding="utf-8")

    monkeypatch.setattr(module, "REPO_ROOT", fake_repo_root)
    resolved_path = module._resolve_prompt_file_path("prompts/paper_small.txt", cfg_path)
    assert resolved_path == prompt_file.resolve()


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
