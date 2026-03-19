"""
文件目的：onefile workflow 编排回归测试。
Module type: General module
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, List
import sys
import json

import pytest
import yaml
from main.evaluation import metrics
from main.watermarking.detect import orchestrator as detect_orchestrator
from main.watermarking.detect.orchestrator import load_scores_for_calibration


def _load_onefile_module(repo_root: Path):
    """
    功能：动态加载 onefile workflow 脚本模块。 

    Dynamically load scripts/run_onefile_workflow.py as module.

    Args:
        repo_root: Repository root path.

    Returns:
        Imported module object.

    Raises:
        RuntimeError: If module spec cannot be loaded.
    """
    module_path = repo_root / "scripts" / "run_onefile_workflow.py"
    spec = importlib.util.spec_from_file_location("run_onefile_workflow", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_onefile_workflow_builds_commands_and_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    功能：验证 onefile 命令顺序与 fail-fast。 

    Validate command sequence and fail-fast behavior for onefile workflow.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    cfg_path = repo_root / "configs" / "default.yaml"

    calls: List[List[str]] = []

    class _Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def _fake_run_ok(cmd: List[str], cwd: str, check: bool, env: dict) -> Any:
        _ = cwd
        _ = check
        _ = env
        calls.append(list(cmd))
        return _Completed(0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run_ok)

    return_code = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="cpu_smoke",
        signoff_profile="baseline",
        dry_run=False,
    )
    assert return_code == 0

    expected_order = [
        "main.cli.run_embed",
        "main.cli.run_detect",
        "main.cli.run_calibrate",
        "main.cli.run_evaluate",
        "run_all_audits.py",
        "run_all_audits.py",
        "run_freeze_signoff.py",
    ]

    observed_order = []
    for command in calls:
        command_text = " ".join(command)
        if "-m main.cli.run_embed" in command_text:
            observed_order.append("main.cli.run_embed")
        elif "-m main.cli.run_detect" in command_text:
            observed_order.append("main.cli.run_detect")
        elif "-m main.cli.run_calibrate" in command_text:
            observed_order.append("main.cli.run_calibrate")
        elif "-m main.cli.run_evaluate" in command_text:
            observed_order.append("main.cli.run_evaluate")
        elif "run_all_audits.py" in command_text and "--strict" not in command:
            observed_order.append("run_all_audits.py")
        elif "run_all_audits.py" in command_text and "--strict" in command:
            observed_order.append("run_all_audits.py")
        elif "run_freeze_signoff.py" in command_text:
            observed_order.append("run_freeze_signoff.py")

    assert observed_order == expected_order
    assert any("--strict" in command for command in calls), "strict 审计步骤必须存在"
    audit_commands = [command for command in calls if "run_all_audits.py" in " ".join(command)]
    assert len(audit_commands) == 2, "must execute audits and audits_strict"
    for audit_command in audit_commands:
        assert "--run-root" in audit_command, "audits command must bind current run_root"
        run_root_arg_idx = audit_command.index("--run-root")
        assert audit_command[run_root_arg_idx + 1] == str(run_root)
    detect_commands = [command for command in calls if "-m" in command and "main.cli.run_detect" in command]
    assert detect_commands, "detect command must be present"
    assert "allow_threshold_fallback_for_tests=true" in detect_commands[0]

    calls_fail_fast: List[List[str]] = []

    def _fake_run_fail_on_calibrate(cmd: List[str], cwd: str, check: bool, env: dict) -> Any:
        _ = cwd
        _ = check
        _ = env
        calls_fail_fast.append(list(cmd))
        command_text = " ".join(cmd)
        if "-m main.cli.run_calibrate" in command_text:
            return _Completed(9)
        return _Completed(0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run_fail_on_calibrate)
    return_code_fail = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="cpu_smoke",
        signoff_profile="baseline",
        dry_run=False,
    )
    assert return_code_fail == 9
    joined_fail_fast = [" ".join(command) for command in calls_fail_fast]
    assert any("main.cli.run_calibrate" in item for item in joined_fail_fast)
    assert not any("main.cli.run_evaluate" in item for item in joined_fail_fast)
    assert not any("run_all_audits.py" in item for item in joined_fail_fast)
    assert not any("run_freeze_signoff.py" in item for item in joined_fail_fast)


def test_onefile_workflow_requires_run_root_or_generates_one(tmp_path: Path) -> None:
    """
    功能：验证 run_root 缺省自动生成与路径约束。 

    Validate generated run_root format and ensure it stays under repo outputs.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    module_path = Path(__file__).resolve().parent.parent / "scripts" / "run_onefile_workflow.py"
    spec = importlib.util.spec_from_file_location("run_onefile_workflow_for_path", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load onefile module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    generated = module._build_run_root(repo_root, None, "cpu_smoke")
    generated.resolve().relative_to((repo_root / "outputs").resolve())
    assert generated.name.startswith("onefile_cpu_smoke_")

    provided_relative = module._build_run_root(repo_root, "outputs/custom_run", "cpu_smoke")
    assert provided_relative == (repo_root / "outputs" / "custom_run").resolve()


def test_onefile_workflow_paper_full_profile_generates_real_sd3_config(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda profile 的真实 SD3 配置与命令覆盖。 

    Validate paper_full_cuda profile writes real SD3 config and command overrides.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_full_run"
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"
    cfg_obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg_obj, dict):
        raise RuntimeError("paper_full_cuda config must be mapping")
    local_model_path = tmp_path / "inspyrenet_plus_ultra.pth"
    local_model_path.write_bytes(b"unit-test-model")
    mask_cfg = cfg_obj.get("mask") if isinstance(cfg_obj.get("mask"), dict) else {}
    mask_cfg["semantic_model_path"] = str(local_model_path)
    cfg_obj["mask"] = mask_cfg
    local_cfg_path = tmp_path / "paper_full_cuda_local.yaml"
    local_cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    profile_cfg_path = module._prepare_profile_cfg_path("paper_full_cuda", run_root, local_cfg_path)
    profile_cfg_text = profile_cfg_path.read_text(encoding="utf-8")
    profile_cfg_obj = yaml.safe_load(profile_cfg_text)

    assert profile_cfg_obj["impl"]["sync_module_id"] == "geometry_latent_sync_sd3"
    assert profile_cfg_obj["impl"]["geometry_extractor_id"] == "attention_anchor_extractor"
    assert profile_cfg_obj["mask"]["semantic_model_path"] == str(local_model_path.resolve())
    assert profile_cfg_obj["embed"]["geometry"]["sync_strength"] == 0.2
    assert profile_cfg_obj["watermark"]["hf"]["tail_truncation_mode"] == cfg_obj["watermark"]["hf"]["tail_truncation_mode"]
    assert profile_cfg_obj["watermark"]["hf"]["selection"] == cfg_obj["watermark"]["hf"]["selection"]
    assert profile_cfg_obj["watermark"]["subspace"]["rank"] == 128
    assert "device: cuda" in profile_cfg_text
    assert "enabled: true" in profile_cfg_text
    assert "alignment_check: true" in profile_cfg_text
    assert "coding_mode: pseudogaussian_template_additive" in profile_cfg_text

    steps = module.build_workflow_steps(
        run_root=run_root,
        cfg_path=profile_cfg_path,
        repo_root=repo_root,
        profile="paper_full_cuda",
        signoff_profile="baseline",
    )
    embed_command_text = " ".join(steps[0].command)
    assert "enable_paper_faithfulness=true" in embed_command_text
    assert "enable_trace_tap=true" in embed_command_text
    assert "force_cpu=\"cpu\"" not in embed_command_text
    step_names = [item.name for item in steps]
    assert "experiment_matrix" in step_names
    assert "multi_protocol_evaluation" not in step_names
    assert "assert_paper_mechanisms" not in step_names


def test_experiment_matrix_cfg_disables_attestation_without_polluting_main_cfg(tmp_path: Path) -> None:
    """
    功能：验证 experiment_matrix 专用 cfg 仅在 matrix 子运行内关闭 paper 与 attestation。

    Verify matrix-specific config disables paper faithfulness and attestation
    only for experiment_matrix sub-runs, without mutating the source cfg.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    cfg_obj = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True,
        },
        "attestation": {
            "enabled": True,
            "require_signed_bundle_verification": True,
        },
        "input_image_path": "top_level_should_be_removed.png",
        "embed": {
            "input_image_path": "main_workflow_default_embed_input.png",
            "preview_generation": {
                "enabled": True,
            },
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    matrix_cfg_path = module._prepare_experiment_matrix_cfg_path(
        "paper_full_cuda",
        run_root,
        cfg_path,
    )

    matrix_cfg = yaml.safe_load(matrix_cfg_path.read_text(encoding="utf-8"))
    original_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert matrix_cfg_path != cfg_path
    assert matrix_cfg["paper_faithfulness"]["enabled"] is False
    assert matrix_cfg["paper_faithfulness"]["alignment_check"] is False
    assert matrix_cfg["attestation"]["enabled"] is False
    assert matrix_cfg["attestation"]["require_signed_bundle_verification"] is False
    assert "input_image_path" not in matrix_cfg
    assert "input_image_path" not in matrix_cfg["embed"]

    assert original_cfg["paper_faithfulness"]["enabled"] is True
    assert original_cfg["paper_faithfulness"]["alignment_check"] is True
    assert original_cfg["attestation"]["enabled"] is True
    assert original_cfg["attestation"]["require_signed_bundle_verification"] is True
    assert original_cfg["input_image_path"] == "top_level_should_be_removed.png"
    assert original_cfg["embed"]["input_image_path"] == "main_workflow_default_embed_input.png"


def test_onefile_workflow_dry_run_skips_dual_branch_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 dry-run 模式不会执行 dual-branch 子流程。 

    Verify dry-run mode does not invoke dual-branch embed/detect execution.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_dry_run"
    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"
    dual_branch_calls = {"count": 0}

    def _forbid_dual_branch(*args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        dual_branch_calls["count"] += 1
        raise AssertionError("dual-branch should not run in dry-run")

    monkeypatch.setattr(module, "_run_dual_branch_embedding_and_detection", _forbid_dual_branch)

    exit_code = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
        signoff_profile="paper",
        dry_run=True,
    )

    assert exit_code == 0
    assert dual_branch_calls["count"] == 0


def test_dual_branch_negative_cfg_disables_attestation_without_polluting_main_cfg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 dual-branch negative branch 专用 cfg 仅在 branch_neg 内关闭 paper 与 attestation。

    Verify branch_neg config disables paper faithfulness and attestation only
    for the negative branch, without mutating the source cfg.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    input_image_path = tmp_path / "clean_input.png"
    input_image_path.write_bytes(b"png-bytes")
    (run_root / "records" / "embed_record.json").write_text(
        json.dumps({"image_path": str(input_image_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg_obj = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True,
        },
        "attestation": {
            "enabled": True,
            "require_signed_bundle_verification": True,
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        cmd_out = Path(cmd[cmd.index("--out") + 1])
        records_dir = cmd_out / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        (records_dir / "detect_record.json").write_text(
            json.dumps({"status": "ok"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)

    branch_neg_root, _ = module._run_dual_branch_embedding_and_detection(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
    )

    branch_cfg_path = branch_neg_root / "artifacts" / "workflow_cfg" / "branch_neg_profile.yaml"
    branch_cfg = yaml.safe_load(branch_cfg_path.read_text(encoding="utf-8"))
    original_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert branch_cfg["paper_faithfulness"]["enabled"] is False
    assert branch_cfg["paper_faithfulness"]["alignment_check"] is False
    assert branch_cfg["attestation"]["enabled"] is False
    assert branch_cfg["attestation"]["require_signed_bundle_verification"] is False

    assert original_cfg["paper_faithfulness"]["enabled"] is True
    assert original_cfg["paper_faithfulness"]["alignment_check"] is True
    assert original_cfg["attestation"]["enabled"] is True
    assert original_cfg["attestation"]["require_signed_bundle_verification"] is True


def test_dual_branch_negative_cfg_preserves_attestation_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证并行 attestation 统计链请求时，negative branch 会保留 attestation 配置与主路径 attestation 载荷。

    Verify negative branch preserves attestation config and embed payload when
    parallel attestation statistics require formal negative samples.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    input_image_path = tmp_path / "clean_input.png"
    input_image_path.write_bytes(b"png-bytes")
    (run_root / "records" / "embed_record.json").write_text(
        json.dumps(
            {
                "image_path": str(input_image_path),
                "attestation": {
                    "status": "ok",
                    "statement": {"schema": "gen_attest_v1"},
                    "signed_bundle": {"schema": "gen_attest_bundle_v1"},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg_obj = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True,
        },
        "attestation": {
            "enabled": True,
            "require_signed_bundle_verification": True,
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        cmd_out = Path(cmd[cmd.index("--out") + 1])
        records_dir = cmd_out / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        (records_dir / "detect_record.json").write_text(
            json.dumps({"status": "ok"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)

    branch_neg_root, _ = module._run_dual_branch_embedding_and_detection(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
        preserve_attestation=True,
    )

    branch_cfg_path = branch_neg_root / "artifacts" / "workflow_cfg" / "branch_neg_profile.yaml"
    branch_cfg = yaml.safe_load(branch_cfg_path.read_text(encoding="utf-8"))
    negative_embed_record = json.loads((branch_neg_root / "records" / "embed_record.json").read_text(encoding="utf-8"))

    assert branch_cfg["attestation"]["enabled"] is True
    assert branch_cfg["attestation"]["require_signed_bundle_verification"] is True
    assert negative_embed_record.get("attestation") is None
    assert negative_embed_record.get("attestation_statement") is None
    assert negative_embed_record.get("attestation_bundle") is None
    provenance_payload = negative_embed_record.get("negative_branch_source_attestation_provenance")
    assert isinstance(provenance_payload, dict)
    assert provenance_payload["statement"]["schema"] == "gen_attest_v1"


def test_negative_branch_attestation_provenance_is_not_consumed_by_formal_payload_extractor() -> None:
    """
    功能：验证 negative branch provenance 不会被 formal attestation payload 提取器直接消费。 

    Verify negative-branch attestation provenance is invisible to the formal
    attestation payload extractor.
    """
    input_record = {
        "is_watermarked": False,
        "negative_branch_source_attestation_provenance": {
            "statement": {"schema": "gen_attest_v1"},
            "attestation_digest": "a" * 64,
            "event_binding_digest": "b" * 64,
        },
    }

    formal_payload = detect_orchestrator._extract_input_attestation_payload(input_record)

    assert formal_payload is None


def test_negative_branch_statement_only_provenance_keeps_image_evidence_but_blocks_event_attestation() -> None:
    """
    功能：验证 negative branch 的 statement-only provenance 仅保留 image evidence，不会形成 event attestation。 

    Verify negative-branch statement-only provenance preserves image evidence
    while forcing the event-level verdict to remain unattested.
    """
    cfg = {
        "attestation": {
            "enabled": True,
            "decision_mode": "content_primary_geo_rescue",
            "lf_weight": 0.5,
            "hf_weight": 0.3,
            "geo_weight": 0.2,
            "threshold": 0.65,
        },
        "__attestation_verify_k_master__": "5" * 64,
    }
    input_record = {
        "is_watermarked": False,
        "negative_branch_source_attestation_provenance": {
            "statement": {
                "schema": "gen_attest_v1",
                "model_id": "m",
                "prompt_commit": "1",
                "seed_commit": "2",
                "plan_digest": "3" * 21 + "3",
                "event_nonce": "4",
                "time_bucket": "2026-03-16T00",
            },
            "attestation_digest": "a" * 64,
            "event_binding_digest": "b" * 64,
        },
    }
    content_evidence_payload = {
        "lf_score": 0.92,
        "score_parts": {
            "hf_attestation_values": [1.0, 0.5, 0.25, 0.125],
        },
        "hf_evidence_summary": {"hf_status": "ok"},
    }

    attestation_context = detect_orchestrator._prepare_detect_attestation_context(cfg, input_record)
    result = detect_orchestrator._build_detect_attestation_result(
        cfg=cfg,
        attestation_context=attestation_context,
        content_evidence_payload=content_evidence_payload,
        geometry_evidence_payload=None,
    )

    assert attestation_context["attestation_source"] == "negative_branch_statement_only_provenance"
    assert attestation_context["authenticity_status"] == "statement_only"
    assert result.get("attestation_source") == "negative_branch_statement_only_provenance"
    authenticity_result = result.get("authenticity_result")
    assert isinstance(authenticity_result, dict)
    assert authenticity_result.get("bundle_status") == "statement_only_provenance_no_bundle"
    image_evidence_result = result.get("image_evidence_result")
    assert isinstance(image_evidence_result, dict)
    assert image_evidence_result.get("status") == "ok"
    assert isinstance(image_evidence_result.get("content_attestation_score"), float)
    final_decision = result.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert final_decision.get("status") == "absent"
    assert final_decision.get("is_event_attested") is False
    assert final_decision.get("event_attestation_score") == pytest.approx(0.0)


def test_negative_branch_statement_only_provenance_supports_low_event_attestation_statistics() -> None:
    """
    功能：验证 statement-only negative 可作为 event_attestation_score 正式低值样本进入校准。 

    Verify statement-only negatives remain valid formal low-score samples for
    event_attestation_score calibration.
    """
    records = [
        {
            "label": True,
            "attestation": {
                "final_event_attested_decision": {
                    "status": "unattested",
                    "is_event_attested": False,
                    "authenticity_status": "authentic",
                    "event_attestation_score": 0.0,
                    "event_attestation_score_name": "event_attestation_score",
                    "event_attestation_statistics_score": 0.81,
                    "event_attestation_statistics_score_name": "event_attestation_statistics_score",
                }
            },
        },
        {
            "label": False,
            "attestation": {
                "image_evidence_result": {
                    "status": "ok",
                    "content_attestation_score": 0.79,
                    "content_attestation_score_name": "content_attestation_score",
                },
                "final_event_attested_decision": {
                    "status": "absent",
                    "is_event_attested": False,
                    "event_attestation_score": 0.0,
                    "event_attestation_score_name": "event_attestation_score",
                    "authenticity_status": "statement_only",
                    "event_attestation_statistics_score": 0.0,
                    "event_attestation_statistics_score_name": "event_attestation_statistics_score",
                },
            },
            "content_evidence_payload": {"status": "ok", "detect_hf_score": 0.95},
        },
    ]

    scores, strata = load_scores_for_calibration(records, score_name="event_attestation_statistics_score")

    assert scores == [pytest.approx(0.0)]
    assert strata["global"]["n_valid"] == 1


def test_dual_branch_negative_embed_record_preserves_formal_plan_anchors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 dual-branch negative embed record 会保留 detect formal path 所需的最小计划锚点。

    Verify negative embed record preserves the minimum embed-time plan anchors
    required by detect formal-path closure.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_run"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    input_image_path = tmp_path / "clean_input.png"
    input_image_path.write_bytes(b"png-bytes")
    source_embed_record = {
        "image_path": str(input_image_path),
        "plan_digest": "a" * 64,
        "cfg_digest": "b" * 64,
        "basis_digest": "c" * 64,
        "plan_input_digest": "f" * 64,
        "plan_input_schema_version": "v1",
        "seed": 7,
        "subspace_planner_impl_identity": {
            "impl_id": "subspace_planner",
            "impl_version": "v1",
            "impl_digest": "d" * 64,
        },
        "subspace_plan": {
            "planner_input_digest": "f" * 64,
            "verifiable_input_domain_spec": {
                "planner_input_digest": "f" * 64,
            },
        },
        "content_evidence": {
            "plan_digest": "a" * 64,
            "mask_digest": "e" * 64,
            "injection_site_spec": {
                "injection_rule_summary": {
                    "plan_digest": "a" * 64,
                }
            },
        },
        "embed_trace": {
            "plan_digest": "a" * 64,
            "injection_evidence": {
                "plan_digest": "a" * 64,
            },
        },
    }
    (run_root / "records" / "embed_record.json").write_text(
        json.dumps(source_embed_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg_obj = {
        "paper_faithfulness": {
            "enabled": True,
            "alignment_check": True,
        },
        "attestation": {
            "enabled": False,
            "require_signed_bundle_verification": False,
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        cmd_out = Path(cmd[cmd.index("--out") + 1])
        records_dir = cmd_out / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        (records_dir / "detect_record.json").write_text(
            json.dumps({"status": "ok"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)

    branch_neg_root, _ = module._run_dual_branch_embedding_and_detection(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
        preserve_attestation=False,
    )

    negative_embed_record = json.loads((branch_neg_root / "records" / "embed_record.json").read_text(encoding="utf-8"))

    assert negative_embed_record["plan_digest"] == "a" * 64
    assert negative_embed_record["cfg_digest"] == "b" * 64
    assert negative_embed_record["basis_digest"] == "c" * 64
    assert negative_embed_record["plan_input_digest"] == "f" * 64
    assert negative_embed_record["plan_input_schema_version"] == "v1"
    assert negative_embed_record["seed"] == 7
    assert negative_embed_record["subspace_planner_impl_identity"]["impl_id"] == "subspace_planner"
    assert negative_embed_record["subspace_plan"]["planner_input_digest"] == "f" * 64
    assert "content_evidence" not in negative_embed_record
    assert "embed_trace" not in negative_embed_record
    assert "injection_evidence" not in negative_embed_record


def test_prepare_detect_records_with_event_attestation_score_rejects_detect_hf_recovery(tmp_path: Path) -> None:
    """
    功能：验证 attestation 统计链不会从 detect_hf_score 恢复正式负样本。

    Verify event-attestation statistics do not recover formal negatives from
    detect_hf_score.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    branch_neg_detect_path = tmp_path / "branch_neg_detect_record.json"
    branch_neg_detect_path.write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {
                    "status": "absent",
                    "score": None,
                    "detect_hf_score": 0.37,
                },
                "attestation": {
                    "image_evidence_result": {"status": "absent", "content_attestation_score": None},
                    "final_event_attested_decision": {
                        "status": "absent",
                        "is_event_attested": False,
                        "event_attestation_score": None,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
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
        pair_count=1,
        branch_neg_detect_record=branch_neg_detect_path,
        score_name="event_attestation_score",
    )

    generated_positive = Path(glob_pattern.replace("*", "positive"))
    generated_negative = Path(glob_pattern.replace("*", "negative"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    negative_attestation = negative_payload.get("attestation")
    assert isinstance(negative_attestation, dict)
    assert negative_attestation["final_event_attested_decision"]["event_attestation_score"] is None

    with pytest.raises(ValueError, match="event_attestation_score"):
        load_scores_for_calibration(
            [positive_payload, negative_payload],
            cfg={
                "calibration": {
                    "exclude_formal_sidecar_disabled_marker": True,
                    "exclude_synthetic_negative_closure_marker": True,
                }
            },
            score_name="event_attestation_score",
        )


def test_prepare_detect_records_with_event_attestation_score_preserves_dual_branch_negative_record(tmp_path: Path) -> None:
    """
    功能：验证 attestation 统计链会透传 dual-branch negative detect record 的正式 attestation 结果。

    Verify event-attestation statistics preserve the real dual-branch negative
    detect-record attestation result without rewriting it to synthetic absent.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    branch_neg_detect_path = tmp_path / "branch_neg_detect_record.json"
    branch_neg_detect_path.write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.14,
                    "detect_hf_score": 0.88,
                },
                "attestation": {
                    "attestation_source": "negative_branch_statement_only_provenance",
                    "authenticity_result": {
                        "status": "statement_only",
                        "bundle_status": "statement_only_provenance_no_bundle",
                        "statement_status": "parsed",
                    },
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.14,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
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
        pair_count=1,
        branch_neg_detect_record=branch_neg_detect_path,
        score_name="event_attestation_score",
    )

    generated_positive = Path(glob_pattern.replace("*", "positive"))
    generated_negative = Path(glob_pattern.replace("*", "negative"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    negative_attestation = negative_payload.get("attestation")
    assert isinstance(negative_attestation, dict)
    assert negative_attestation["attestation_source"] == "negative_branch_statement_only_provenance"
    assert negative_attestation["authenticity_result"]["bundle_status"] == "statement_only_provenance_no_bundle"
    assert negative_attestation["image_evidence_result"]["status"] == "ok"
    assert negative_attestation["image_evidence_result"]["content_attestation_score"] == pytest.approx(0.14)
    assert negative_attestation["final_event_attested_decision"]["status"] == "unattested"
    assert negative_attestation["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.0)
    assert negative_payload["ground_truth_source"] == "dual_branch_negative"
    assert negative_payload["content_evidence_payload"].get("attestation_clone_fallback_status") is None

    scores, strata = load_scores_for_calibration(
        [positive_payload, negative_payload],
        cfg={
            "calibration": {
                "exclude_formal_sidecar_disabled_marker": True,
                "exclude_synthetic_negative_closure_marker": True,
            }
        },
        score_name="event_attestation_score",
    )

    assert scores == [pytest.approx(0.0)]
    assert strata["global"]["n_valid"] == 1


def test_prepare_detect_records_with_content_score_preserves_dual_branch_negative_record_without_synthetic_closure(
    tmp_path: Path,
) -> None:
    """
    功能：验证 real dual-branch negative detect 成功时，content_score 主链不会退回 synthetic closure。 

    Verify content_score calibration keeps the real dual-branch negative sample
    and avoids synthetic closure markers when branch-neg detect succeeds.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "content_evidence_payload": {"status": "ok", "score": 0.91},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    branch_neg_detect_path = tmp_path / "branch_neg_detect_record.json"
    branch_neg_detect_path.write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.14,
                },
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
        pair_count=1,
        branch_neg_detect_record=branch_neg_detect_path,
        score_name="content_score",
    )

    generated_positive = Path(glob_pattern.replace("*", "positive"))
    generated_negative = Path(glob_pattern.replace("*", "negative"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    negative_content = negative_payload.get("content_evidence_payload")
    assert isinstance(negative_content, dict)
    assert negative_payload["ground_truth_source"] == "dual_branch_negative"
    assert negative_content["score"] == pytest.approx(0.14)
    assert negative_content.get("calibration_sample_usage") != "synthetic_negative_for_ground_truth_closure"
    assert negative_content.get("calibration_sample_origin") != "synthetic_negative_bundle"
    assert negative_content.get("attestation_clone_fallback_status") is None

    scores, strata = load_scores_for_calibration(
        [positive_payload, negative_payload],
        cfg={
            "calibration": {
                "exclude_formal_sidecar_disabled_marker": True,
                "exclude_synthetic_negative_closure_marker": True,
            }
        },
        score_name="content_score",
    )

    assert scores == [pytest.approx(0.14)]
    assert strata["global"]["n_valid"] == 1


def test_prepare_detect_records_with_content_attestation_score_clone_fallback_marks_attestation_unavailable(
    tmp_path: Path,
) -> None:
    """
    功能：验证 content_attestation_score 的 clone fallback 会将 attestation 标记为 unavailable。 

    Verify clone fallback for content_attestation_score marks the copied
    attestation result as unavailable instead of inheriting the source formal
    verdict.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
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
        pair_count=1,
        branch_neg_detect_record=tmp_path / "missing_branch_neg_detect_record.json",
        score_name="content_attestation_score",
        dual_branch_failure_reason="RuntimeError: branch negative missing",
    )

    generated_positive = Path(glob_pattern.replace("*", "positive"))
    generated_negative = Path(glob_pattern.replace("*", "negative"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    negative_content = negative_payload.get("content_evidence_payload")
    assert isinstance(negative_content, dict)
    assert negative_content.get("calibration_sample_usage") is None
    assert negative_content.get("calibration_sample_origin") is None
    assert negative_content.get("dual_branch_failure_reason") == "RuntimeError: branch negative missing"

    negative_attestation = negative_payload.get("attestation")
    assert isinstance(negative_attestation, dict)
    image_evidence_result = negative_attestation.get("image_evidence_result")
    assert isinstance(image_evidence_result, dict)
    assert negative_attestation["status"] == "absent"
    assert negative_attestation["attestation_absent_reason"] == "attestation_unavailable_in_clone_fallback"
    assert image_evidence_result["status"] == "absent"
    assert image_evidence_result["content_attestation_score"] is None
    assert image_evidence_result["attestation_unavailable_reason"] == "attestation_unavailable_in_clone_fallback"
    assert negative_content["attestation_clone_fallback_status"] == "attestation_unavailable_in_clone_fallback"

    with pytest.raises(ValueError, match="content_attestation_score"):
        load_scores_for_calibration(
            [positive_payload, negative_payload],
            cfg={
                "calibration": {
                    "exclude_formal_sidecar_disabled_marker": True,
                    "exclude_synthetic_negative_closure_marker": True,
                }
            },
            score_name="content_attestation_score",
        )


def test_prepare_detect_records_with_event_attestation_score_clone_fallback_marks_attestation_unavailable(
    tmp_path: Path,
) -> None:
    """
    功能：验证 event_attestation_score 的 clone fallback 会将 attestation 标记为 unavailable。 

    Verify clone fallback for event_attestation_score marks the copied
    attestation verdict as unavailable instead of inheriting the source formal
    event-attested result.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
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
        pair_count=1,
        branch_neg_detect_record=tmp_path / "missing_branch_neg_detect_record.json",
        score_name="event_attestation_score",
        dual_branch_failure_reason="RuntimeError: branch negative missing",
    )

    generated_negative = Path(glob_pattern.replace("*", "negative"))
    generated_positive = Path(glob_pattern.replace("*", "positive"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    negative_content = negative_payload.get("content_evidence_payload")
    assert isinstance(negative_content, dict)
    assert negative_content.get("calibration_sample_usage") is None
    assert negative_content.get("calibration_sample_origin") is None
    assert negative_content.get("dual_branch_failure_reason") == "RuntimeError: branch negative missing"

    negative_attestation = negative_payload.get("attestation")
    assert isinstance(negative_attestation, dict)
    final_decision = negative_attestation.get("final_event_attested_decision")
    assert isinstance(final_decision, dict)
    assert negative_attestation["status"] == "absent"
    assert negative_attestation["attestation_absent_reason"] == "attestation_unavailable_in_clone_fallback"
    assert final_decision["status"] == "absent"
    assert final_decision["is_event_attested"] is False
    assert final_decision["event_attestation_score"] is None
    assert final_decision["attestation_unavailable_reason"] == "attestation_unavailable_in_clone_fallback"
    image_evidence_result = negative_attestation.get("image_evidence_result")
    assert isinstance(image_evidence_result, dict)
    assert image_evidence_result["content_attestation_score"] is None
    assert negative_content["attestation_clone_fallback_status"] == "attestation_unavailable_in_clone_fallback"

    with pytest.raises(ValueError, match="event_attestation_score"):
        load_scores_for_calibration(
            [positive_payload, negative_payload],
            cfg={
                "calibration": {
                    "exclude_formal_sidecar_disabled_marker": True,
                    "exclude_synthetic_negative_closure_marker": True,
                }
            },
            score_name="event_attestation_score",
        )


def test_prepare_detect_records_with_event_attestation_score_clone_fallback_restores_attack_metadata(
    tmp_path: Path,
) -> None:
    """
    功能：验证 event_attestation_score 的 clone fallback 仅补写 attack metadata，不污染 attestation 子树。

    Verify clone fallback for event_attestation_score restores trusted attack
    grouping metadata while keeping formal attestation semantics unchanged.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attestation": {
                    "attestation_source": "formal_input_payload",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix_detect_path = (
        run_root
        / "outputs"
        / "experiment_matrix"
        / "experiments"
        / "item_0000"
        / "artifacts"
        / "evaluate_inputs"
        / "detect_record_with_attack.json"
    )
    matrix_detect_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attack_family": "jpeg",
                "attack_params_version": "p1",
                "attack": {"family": "jpeg", "params_version": "p1"},
                "attestation": {
                    "status": "absent",
                    "attestation_absent_reason": "attestation_disabled",
                    "final_event_attested_decision": {
                        "status": "absent",
                        "is_event_attested": False,
                    },
                },
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
        pair_count=1,
        branch_neg_detect_record=tmp_path / "missing_branch_neg_detect_record.json",
        score_name="event_attestation_score",
        dual_branch_failure_reason="RuntimeError: branch negative missing",
    )

    generated_negative = Path(glob_pattern.replace("*", "negative"))
    generated_positive = Path(glob_pattern.replace("*", "positive"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    assert positive_payload["attack_family"] == "jpeg"
    assert positive_payload["attack_params_version"] == "p1"
    assert positive_payload["attack"]["family"] == "jpeg"
    assert negative_payload["attack_family"] == "jpeg"
    assert negative_payload["attack_params_version"] == "p1"
    assert negative_payload["attack"]["family"] == "jpeg"

    protocol_spec = {
        "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
        "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
    }
    assert metrics.build_attack_group_key(positive_payload, protocol_spec) == "jpeg::p1"
    assert metrics.build_attack_group_key(negative_payload, protocol_spec) == "jpeg::p1"

    assert positive_payload["attestation"]["attestation_source"] == "formal_input_payload"
    assert positive_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.93)
    assert negative_payload["attestation"]["status"] == "absent"
    assert negative_payload["attestation"]["attestation_absent_reason"] == "attestation_unavailable_in_clone_fallback"
    assert negative_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] is None


def test_prepare_detect_records_with_content_score_clone_fallback_keeps_synthetic_negative_closure(
    tmp_path: Path,
) -> None:
    """
    功能：验证 content_score 主链的 clone fallback 继续保留 synthetic negative closure。 

    Verify clone fallback for content_score keeps the existing synthetic
    negative closure behavior unchanged.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "content_evidence_payload": {"status": "ok", "score": 0.91},
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
        pair_count=1,
        branch_neg_detect_record=tmp_path / "missing_branch_neg_detect_record.json",
        score_name="content_score",
        dual_branch_failure_reason="RuntimeError: branch negative missing",
    )

    generated_negative = Path(glob_pattern.replace("*", "negative"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))
    negative_content = negative_payload.get("content_evidence_payload")
    assert isinstance(negative_content, dict)
    assert negative_content["status"] == "ok"
    assert negative_content["score"] == pytest.approx(-1.0)
    assert negative_content["calibration_sample_usage"] == "synthetic_negative_for_ground_truth_closure"
    assert negative_content["calibration_sample_origin"] == "synthetic_negative_bundle"
    assert negative_content["dual_branch_failure_reason"] == "RuntimeError: branch negative missing"


def test_prepare_detect_records_with_content_score_clone_fallback_restores_attack_metadata(
    tmp_path: Path,
) -> None:
    """
    功能：验证 content_score 的 clone fallback 也可保留可信 attack grouping metadata。

    Verify clone fallback for content_score restores trusted attack grouping
    metadata while keeping synthetic negative closure behavior unchanged.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "content_evidence_payload": {"status": "ok", "score": 0.91},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix_detect_path = (
        run_root
        / "outputs"
        / "experiment_matrix"
        / "experiments"
        / "item_0000"
        / "artifacts"
        / "evaluate_inputs"
        / "detect_record_with_attack.json"
    )
    matrix_detect_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attack_family": "jpeg",
                "attack_params_version": "p1",
                "attack": {"family": "jpeg", "params_version": "p1"},
                "attestation": {
                    "status": "absent",
                    "attestation_absent_reason": "attestation_disabled",
                },
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
        pair_count=1,
        branch_neg_detect_record=tmp_path / "missing_branch_neg_detect_record.json",
        score_name="content_score",
        dual_branch_failure_reason="RuntimeError: branch negative missing",
    )

    generated_negative = Path(glob_pattern.replace("*", "negative"))
    generated_positive = Path(glob_pattern.replace("*", "positive"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    protocol_spec = {
        "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
        "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
    }
    assert metrics.build_attack_group_key(positive_payload, protocol_spec) == "jpeg::p1"
    assert metrics.build_attack_group_key(negative_payload, protocol_spec) == "jpeg::p1"
    assert negative_payload["content_evidence_payload"]["score"] == pytest.approx(-1.0)
    assert negative_payload["content_evidence_payload"]["calibration_sample_usage"] == "synthetic_negative_for_ground_truth_closure"


def test_build_minimal_repro_bundle_excludes_signoff_report_self_reference(tmp_path: Path) -> None:
    """
    功能：验证最小 repro bundle 不绑定 signoff_report 自身，避免 signoff 自循环失配。

    Verify minimal repro bundle excludes signoff_report from pointers so a
    later signoff refresh does not invalidate the bundle itself.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    audit_module_path = repo_root / "scripts" / "audits" / "audit_repro_bundle_integrity.py"
    audit_spec = importlib.util.spec_from_file_location("audit_repro_bundle_integrity", audit_module_path)
    if audit_spec is None or audit_spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {audit_module_path}")
    audit_module = importlib.util.module_from_spec(audit_spec)
    sys.modules[audit_spec.name] = audit_module
    audit_spec.loader.exec_module(audit_module)

    run_root = tmp_path / "run_root"
    (run_root / "artifacts" / "signoff").mkdir(parents=True, exist_ok=True)
    (run_root / "records").mkdir(parents=True, exist_ok=True)

    (run_root / "artifacts" / "run_closure.json").write_text(
        json.dumps(
            {
                "cfg_digest": "cfg-digest",
                "plan_digest": "plan-digest",
                "thresholds_digest": "thresholds-digest",
                "threshold_metadata_digest": "threshold-metadata-digest",
                "impl_digest": "impl-digest",
                "fusion_rule_version": "v1",
                "policy_path": "content_np_geo_rescue",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "records" / "evaluate_record.json").write_text(
        json.dumps(
            {
                "cfg_digest": "cfg-digest",
                "plan_digest": "plan-digest",
                "thresholds_digest": "thresholds-digest",
                "threshold_metadata_digest": "threshold-metadata-digest",
                "impl_digest": "impl-digest",
                "fusion_rule_version": "v1",
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "attack-protocol-digest",
                "policy_path": "content_np_geo_rescue",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "evaluation_report.json").write_text(
        json.dumps(
            {
                "evaluation_report": {
                    "cfg_digest": "cfg-digest",
                    "plan_digest": "plan-digest",
                    "thresholds_digest": "thresholds-digest",
                    "threshold_metadata_digest": "threshold-metadata-digest",
                    "impl_digest": "impl-digest",
                    "fusion_rule_version": "v1",
                    "attack_protocol_version": "attack_protocol_v1",
                    "attack_protocol_digest": "attack-protocol-digest",
                    "policy_path": "content_np_geo_rescue",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    signoff_report_path = run_root / "artifacts" / "signoff" / "signoff_report.json"
    signoff_report_path.write_text(
        json.dumps({"status": "initial"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg_path = repo_root / "configs" / "paper_full_cuda.yaml"

    module._build_minimal_repro_bundle(run_root, cfg_path=cfg_path)

    pointers_path = run_root / "artifacts" / "repro_bundle" / "pointers.json"
    pointers_obj = json.loads(pointers_path.read_text(encoding="utf-8"))
    pointer_paths = [item["path"] for item in pointers_obj["files"]]
    assert "artifacts/signoff/signoff_report.json" not in pointer_paths

    manifest_path = run_root / "artifacts" / "repro_bundle" / "manifest.json"
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_obj["config_path_repo_relative"] == "configs/paper_full_cuda.yaml"
    assert manifest_obj["run_root_relative"] == "."
    assert manifest_obj["paths_relative"]["run_closure"] == "artifacts/run_closure.json"
    assert pointers_obj["run_root_relative"] == "."

    signoff_report_path.write_text(
        json.dumps({"status": "refreshed"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert audit_module.main(str(repo_root), str(run_root)) == 0



def test_parallel_attestation_statistics_workflow_writes_distinct_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 onefile 会在主链之外并行写出独立 attestation 统计链工件与汇总。

    Verify onefile writes distinct parallel attestation statistics artifacts and
    workflow-level summary anchors without overwriting the primary chain.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "branch_neg" / "records").mkdir(parents=True, exist_ok=True)

    cfg_obj = {
        "paper_faithfulness": {"enabled": True, "alignment_check": True},
        "attestation": {"enabled": True, "require_signed_bundle_verification": True},
        "calibration": {"score_name": "content_score", "minimal_ground_truth_pair_count": 1},
        "evaluate": {"score_name": "content_score", "minimal_ground_truth_pair_count": 1},
        "parallel_attestation_statistics": {
            "enabled": True,
            "calibration_score_name": "event_attestation_statistics_score",
            "evaluate_score_name": "event_attestation_statistics_score",
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    (run_root / "records" / "detect_record.json").write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "content_evidence_payload": {"status": "ok", "score": 0.91},
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.89,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "authenticity_status": "authentic",
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                        "event_attestation_statistics_score": 0.89,
                        "event_attestation_statistics_score_name": "event_attestation_statistics_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "branch_neg" / "records" / "detect_record.json").write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {"status": "ok", "score": 0.21},
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.12,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                        "authenticity_status": "statement_only",
                        "event_attestation_statistics_score": 0.0,
                        "event_attestation_statistics_score_name": "event_attestation_statistics_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (run_root / "records" / "calibration_record.json").write_text(
        json.dumps(
            {
                "threshold_id": "content_score_np_fpr_0_01",
                "threshold_key_used": "fpr_0_01",
                "calibration_summary": {
                    "score_name": "content_score",
                    "threshold_value": 0.5,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "thresholds" / "thresholds_artifact.json").write_text(
        json.dumps(
            {
                "threshold_id": "content_score_np_fpr_0_01",
                "score_name": "content_score",
                "threshold_value": 0.5,
                "threshold_key_used": "fpr_0_01",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "records" / "evaluate_record.json").write_text(
        json.dumps(
            {
                "threshold_key_used": "fpr_0_01",
                "metrics": {"score_name": "content_score", "threshold_value": 0.5},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "evaluation_report.json").write_text(
        json.dumps({"evaluation_report": {"score_name": "content_score"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        out_root = Path(cmd[cmd.index("--out") + 1])
        config_path = Path(cmd[cmd.index("--config") + 1])
        stage_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stage_name = "calibration" if "run_calibrate" in " ".join(cmd) else "evaluate"
        score_name = stage_cfg[stage_name]["score_name"]
        (out_root / "records").mkdir(parents=True, exist_ok=True)
        (out_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
        if stage_name == "calibration":
            (out_root / "records" / "calibration_record.json").write_text(
                json.dumps(
                    {
                        "threshold_id": f"{score_name}_np_fpr_0_01",
                        "threshold_key_used": "fpr_0_01",
                        "calibration_summary": {"score_name": score_name, "threshold_value": 0.42},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_root / "artifacts" / "thresholds" / "thresholds_artifact.json").write_text(
                json.dumps(
                    {
                        "threshold_id": f"{score_name}_np_fpr_0_01",
                        "score_name": score_name,
                        "threshold_value": 0.42,
                        "threshold_key_used": "fpr_0_01",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            (out_root / "records" / "evaluate_record.json").write_text(
                json.dumps(
                    {
                        "threshold_key_used": "fpr_0_01",
                        "metrics": {"score_name": score_name, "threshold_value": 0.42},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_root / "artifacts" / "evaluation_report.json").write_text(
                json.dumps({"evaluation_report": {"score_name": score_name}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)

    module._run_parallel_attestation_statistics_workflow(
        repo_root=repo_root,
        run_root=run_root,
        cfg_path=cfg_path,
        profile="paper_full_cuda",
    )

    summary_path = run_root / "artifacts" / "parallel_attestation_statistics_summary.json"
    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    content_chain = summary_obj["content_score_chain"]
    attestation_chain = summary_obj["event_attestation_statistics_score_chain"]

    assert content_chain["score_name"] == "content_score"
    assert attestation_chain["score_name"] == "event_attestation_statistics_score"
    assert content_chain["threshold_id"] == "content_score_np_fpr_0_01"
    assert attestation_chain["threshold_id"] == "event_attestation_statistics_score_np_fpr_0_01"
    assert content_chain["thresholds_artifact_path"] != attestation_chain["thresholds_artifact_path"]
    assert summary_obj["run_root_relative"] == "."
    assert summary_obj["parallel_run_root_relative"] == "outputs/parallel_attestation_statistics"
    assert content_chain["run_root_relative"] == "."
    assert attestation_chain["run_root_relative"] == "outputs/parallel_attestation_statistics"
    assert content_chain["thresholds_artifact_path_relative"] == "artifacts/thresholds/thresholds_artifact.json"
    assert attestation_chain["thresholds_artifact_path_relative"] == (
        "outputs/parallel_attestation_statistics/artifacts/thresholds/thresholds_artifact.json"
    )


def test_ensure_experiment_matrix_grid_summary_anchors_adds_path_views(tmp_path: Path) -> None:
    """
    功能：验证 onefile 的 grid_summary 收口会补齐 batch_root 与 relative path 视图。

    Verify onefile grid_summary closure appends batch-root path views without
    mutating existing anchor semantics.

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
                "total": 1,
                "succeeded": 1,
                "failed": 0,
                "cfg_digest": "cfg-digest",
                "thresholds_digest": "thresholds-digest",
                "threshold_metadata_digest": "threshold-metadata-digest",
                "attack_protocol_version": "attack_protocol_v1",
                "attack_protocol_digest": "attack-protocol-digest",
                "attack_coverage_digest": "attack-coverage-digest",
                "impl_digest": "impl-digest",
                "fusion_rule_version": "v1",
                "policy_path": "content_np_geo_rescue",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ensure_grid_summary = getattr(module, "_ensure_experiment_matrix_grid_summary_anchors")
    ensure_grid_summary(run_root)

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_obj["batch_root_relative"] == "."
    assert summary_obj["summary_path_relative"] == "artifacts/grid_summary.json"
    assert summary_obj["paths_relative"]["summary_path"] == "artifacts/grid_summary.json"


def test_parallel_attestation_statistics_workflow_prefers_experiment_matrix_attack_aware_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 paper 并行 attestation 统计优先消费 experiment_matrix 的 attack-aware 正样本记录。

    Verify paper parallel attestation statistics prefer experiment_matrix
    attack-aware positive records while preserving formal positive/negative
    attestation evidence sources.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "branch_neg" / "records").mkdir(parents=True, exist_ok=True)

    cfg_obj = {
        "calibration": {"score_name": "content_score", "minimal_ground_truth_pair_count": 1},
        "evaluate": {"score_name": "content_score", "minimal_ground_truth_pair_count": 1},
        "parallel_attestation_statistics": {
            "enabled": True,
            "calibration_score_name": "event_attestation_score",
            "evaluate_score_name": "event_attestation_score",
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    (run_root / "records" / "detect_record.json").write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "canonical prompt",
                "content_evidence_payload": {"status": "ok", "score": 0.91},
                "attestation": {
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.89,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.89,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "branch_neg" / "records" / "detect_record.json").write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "inference_prompt": "negative prompt",
                "content_evidence_payload": {"status": "ok", "score": 0.14},
                "attestation": {
                    "attestation_source": "negative_branch_statement_only_provenance",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.14,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix_detect_path = (
        run_root
        / "outputs"
        / "experiment_matrix"
        / "experiments"
        / "item_0000"
        / "artifacts"
        / "evaluate_inputs"
        / "detect_record_with_attack.json"
    )
    matrix_detect_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attack_family": "jpeg",
                "attack_params_version": "p1",
                "attack": {"family": "jpeg", "params_version": "p1"},
                "content_evidence_payload": {"status": "ok", "score": 0.31},
                "attestation": {
                    "attestation_source": "matrix_detect_record_attestation",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.61,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.61,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    observed_stage_names: list[str] = []

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        out_root = Path(cmd[cmd.index("--out") + 1])
        config_path = Path(cmd[cmd.index("--config") + 1])
        stage_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stage_name = "calibration" if "run_calibrate" in " ".join(cmd) else "evaluate"
        observed_stage_names.append(stage_name)
        detect_records_glob = stage_cfg[stage_name]["detect_records_glob"]
        detect_records_path = Path(detect_records_glob)
        detect_records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(detect_records_path.parent.glob(detect_records_path.name))
        ]
        assert len(detect_records) == 2

        positive_payload = next(record for record in detect_records if record.get("label") is True)
        negative_payload = next(record for record in detect_records if record.get("label") is False)

        assert positive_payload["inference_prompt"] == "matrix prompt"
        assert positive_payload["attack"]["family"] == "jpeg"
        assert positive_payload["content_evidence_payload"]["score"] == pytest.approx(0.31)
        assert positive_payload["attestation"]["attestation_source"] == "matrix_detect_record_attestation"
        assert positive_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.61)
        assert positive_payload["attestation"]["image_evidence_result"]["content_attestation_score"] == pytest.approx(0.61)

        assert negative_payload["inference_prompt"] == "matrix prompt"
        assert negative_payload["attack"]["family"] == "jpeg"
        assert negative_payload["content_evidence_payload"]["score"] == pytest.approx(0.14)
        assert negative_payload["attestation"]["attestation_source"] == "negative_branch_statement_only_provenance"
        assert negative_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.0)

        (out_root / "records").mkdir(parents=True, exist_ok=True)
        (out_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
        if stage_name == "calibration":
            (out_root / "records" / "calibration_record.json").write_text(
                json.dumps(
                    {
                        "threshold_id": "event_attestation_score_np_fpr_0_01",
                        "threshold_key_used": "fpr_0_01",
                        "calibration_summary": {
                            "score_name": "event_attestation_score",
                            "threshold_value": 0.42,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_root / "artifacts" / "thresholds" / "thresholds_artifact.json").write_text(
                json.dumps(
                    {
                        "threshold_id": "event_attestation_score_np_fpr_0_01",
                        "score_name": "event_attestation_score",
                        "threshold_value": 0.42,
                        "threshold_key_used": "fpr_0_01",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            (out_root / "records" / "evaluate_record.json").write_text(
                json.dumps(
                    {
                        "threshold_key_used": "fpr_0_01",
                        "metrics": {
                            "score_name": "event_attestation_score",
                            "threshold_value": 0.42,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_root / "artifacts" / "evaluation_report.json").write_text(
                json.dumps({"evaluation_report": {"score_name": "event_attestation_score"}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)

    module._run_parallel_attestation_statistics_workflow(
        repo_root=repo_root,
        run_root=run_root,
        cfg_path=cfg_path,
        profile="paper_full_cuda",
    )

    assert observed_stage_names == ["calibration", "evaluate"]


def test_prepare_detect_records_with_event_attestation_score_restores_attack_metadata_from_matrix_fallback(
    tmp_path: Path,
) -> None:
    """
    功能：验证 dual-branch fallback 可从同次 matrix 输出补写 attack-aware 元数据。 

    Verify dual-branch fallback restores trusted attack grouping metadata from
    same-run matrix outputs without overlaying formal attestation payloads.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attestation": {
                    "attestation_source": "formal_input_payload",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    branch_neg_detect_path = tmp_path / "branch_neg_detect_record.json"
    branch_neg_detect_path.write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.14,
                },
                "attestation": {
                    "attestation_source": "negative_branch_statement_only_provenance",
                    "authenticity_result": {
                        "status": "statement_only",
                        "bundle_status": "statement_only_provenance_no_bundle",
                        "statement_status": "parsed",
                    },
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.14,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix_detect_path = (
        run_root
        / "outputs"
        / "experiment_matrix"
        / "experiments"
        / "item_0000"
        / "artifacts"
        / "evaluate_inputs"
        / "detect_record_with_attack.json"
    )
    matrix_detect_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attack_family": "jpeg",
                "attack_params_version": "p1",
                "attack": {"family": "jpeg", "params_version": "p1"},
                "attestation": {
                    "status": "absent",
                    "attestation_absent_reason": "attestation_disabled",
                    "final_event_attested_decision": {
                        "status": "absent",
                        "is_event_attested": False,
                    },
                },
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
        pair_count=1,
        branch_neg_detect_record=branch_neg_detect_path,
        score_name="event_attestation_score",
    )

    generated_positive = Path(glob_pattern.replace("*", "positive"))
    generated_negative = Path(glob_pattern.replace("*", "negative"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    assert positive_payload["attack_family"] == "jpeg"
    assert positive_payload["attack_params_version"] == "p1"
    assert positive_payload["attack"]["family"] == "jpeg"
    assert negative_payload["attack_family"] == "jpeg"
    assert negative_payload["attack_params_version"] == "p1"
    assert negative_payload["attack"]["family"] == "jpeg"

    assert positive_payload["attestation"]["attestation_source"] == "formal_input_payload"
    assert positive_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.93)
    assert negative_payload["attestation"]["attestation_source"] == "negative_branch_statement_only_provenance"
    assert negative_payload["attestation"]["authenticity_result"]["bundle_status"] == "statement_only_provenance_no_bundle"
    assert negative_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.0)

    protocol_spec = {
        "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
        "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
    }
    assert metrics.build_attack_group_key(positive_payload, protocol_spec) == "jpeg::p1"
    assert metrics.build_attack_group_key(negative_payload, protocol_spec) == "jpeg::p1"


def test_prepare_detect_records_with_event_attestation_score_expands_multi_attack_groups_for_multi_pair_fallback(
    tmp_path: Path,
) -> None:
    """
    功能：当 matrix top-level prompt 缺失且存在多 attack-group 时，fallback 必须按可信 join key 展开。 

    Verify fallback expands trusted attack metadata contexts for multi-pair,
    multi-attack-group runs when matrix records only expose infer_trace prompt anchors.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    canonical_prompt = "matrix canonical prompt"
    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": canonical_prompt,
                "infer_trace": {
                    "inference_prompt": canonical_prompt,
                },
                "content_evidence_payload": {"status": "ok", "score": 0.91},
                "attestation": {
                    "attestation_source": "formal_input_payload",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.93,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.93,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
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
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "infer_trace": {
                    "inference_prompt": canonical_prompt,
                },
                "content_evidence_payload": {"status": "ok", "score": 0.14},
                "attestation": {
                    "attestation_source": "negative_branch_statement_only_provenance",
                    "authenticity_result": {
                        "status": "statement_only",
                        "bundle_status": "statement_only_provenance_no_bundle",
                        "statement_status": "parsed",
                    },
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.14,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix_specs = [
        ("jpeg", "p1"),
        ("rotate", "v1"),
    ]
    for index, (attack_family, attack_params_version) in enumerate(matrix_specs):
        matrix_detect_path = (
            run_root
            / "outputs"
            / "experiment_matrix"
            / "experiments"
            / f"item_{index:04d}"
            / "artifacts"
            / "evaluate_inputs"
            / "detect_record_with_attack.json"
        )
        matrix_detect_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_detect_path.write_text(
            json.dumps(
                {
                    "label": True,
                    "ground_truth": True,
                    "is_watermarked": True,
                    "attack_family": attack_family,
                    "attack_params_version": attack_params_version,
                    "attack": {
                        "family": attack_family,
                        "params_version": attack_params_version,
                    },
                    "attack_metadata_source_prompt": canonical_prompt,
                    "attack_metadata_source_prompt_field": "infer_trace.inference_prompt",
                    "attack_metadata_join_key": module._build_attack_metadata_join_key(
                        canonical_prompt,
                        attack_family,
                        attack_params_version,
                    ),
                    "infer_trace": {
                        "inference_prompt": canonical_prompt,
                    },
                    "attestation": {
                        "status": "absent",
                        "attestation_absent_reason": "attestation_disabled",
                        "final_event_attested_decision": {
                            "status": "absent",
                            "is_event_attested": False,
                        },
                    },
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
        pair_count=2,
        prompts=["gt prompt 1", "gt prompt 2"],
        branch_neg_detect_record=branch_neg_detect_path,
        score_name="event_attestation_score",
    )

    matched_paths = sorted(Path(glob_pattern).parent.glob(Path(glob_pattern).name))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in matched_paths]

    assert len(payloads) == 8

    join_keys = {payload["attack_metadata_join_key"] for payload in payloads}
    assert len(join_keys) == 2

    protocol_spec = {
        "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
        "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
    }

    grouped_payloads: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        grouped_payloads.setdefault(payload["attack_metadata_join_key"], []).append(payload)
        assert payload["attack_metadata_source_prompt"] == canonical_prompt
        assert payload["attack_metadata_source_prompt_field"] == "infer_trace.inference_prompt"
        assert payload["infer_trace"]["inference_prompt"] == canonical_prompt
        assert payload["inference_prompt"] in {"gt prompt 1", "gt prompt 2"}

    assert {metrics.build_attack_group_key(group[0], protocol_spec) for group in grouped_payloads.values()} == {
        "jpeg::p1",
        "rotate::v1",
    }

    for group_payloads in grouped_payloads.values():
        assert len(group_payloads) == 4
        positive_payloads = [payload for payload in group_payloads if payload.get("label") is True]
        negative_payloads = [payload for payload in group_payloads if payload.get("label") is False]
        assert len(positive_payloads) == 2
        assert len(negative_payloads) == 2
        assert {payload["inference_prompt"] for payload in positive_payloads} == {"gt prompt 1", "gt prompt 2"}
        assert {payload["inference_prompt"] for payload in negative_payloads} == {"gt prompt 1", "gt prompt 2"}
        assert all(payload["attestation"]["attestation_source"] == "formal_input_payload" for payload in positive_payloads)
        assert all(
            payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.93)
            for payload in positive_payloads
        )
        assert all(
            payload["attestation"]["attestation_source"] == "negative_branch_statement_only_provenance"
            for payload in negative_payloads
        )
        assert all(
            payload["attestation"]["authenticity_result"]["bundle_status"] == "statement_only_provenance_no_bundle"
            for payload in negative_payloads
        )
        assert all(
            payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.0)
            for payload in negative_payloads
        )


def test_parallel_attestation_statistics_workflow_fallback_preserves_attack_metadata_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 parallel event fallback 仍可保留 attack-aware 分组元数据。 

    Verify parallel event fallback keeps attack grouping metadata from matrix
    context while preserving canonical positive and statement-only negative
    attestation semantics.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "artifacts" / "branch_neg" / "records").mkdir(parents=True, exist_ok=True)

    cfg_obj = {
        "calibration": {
            "score_name": "content_score",
            "minimal_ground_truth_pair_count": 1,
        },
        "evaluate": {
            "score_name": "content_score",
            "minimal_ground_truth_pair_count": 1,
        },
        "parallel_attestation_statistics": {
            "enabled": True,
            "calibration_score_name": "event_attestation_score",
            "evaluate_score_name": "event_attestation_score",
        },
    }
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    (run_root / "records" / "detect_record.json").write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "content_evidence_payload": {"status": "ok", "score": 0.91},
                "attestation": {
                    "attestation_source": "formal_input_payload",
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.89,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "attested",
                        "is_event_attested": True,
                        "event_attestation_score": 0.89,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "branch_neg" / "records" / "detect_record.json").write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {"status": "ok", "score": 0.14},
                "attestation": {
                    "attestation_source": "negative_branch_statement_only_provenance",
                    "authenticity_result": {
                        "status": "statement_only",
                        "bundle_status": "statement_only_provenance_no_bundle",
                        "statement_status": "parsed",
                    },
                    "image_evidence_result": {
                        "status": "ok",
                        "content_attestation_score": 0.14,
                        "content_attestation_score_name": "content_attestation_score",
                    },
                    "final_event_attested_decision": {
                        "status": "unattested",
                        "is_event_attested": False,
                        "event_attestation_score": 0.0,
                        "event_attestation_score_name": "event_attestation_score",
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    matrix_detect_path = (
        run_root
        / "outputs"
        / "experiment_matrix"
        / "experiments"
        / "item_0000"
        / "artifacts"
        / "evaluate_inputs"
        / "detect_record_with_attack.json"
    )
    matrix_detect_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "inference_prompt": "matrix prompt",
                "attack_family": "jpeg",
                "attack_params_version": "p1",
                "attack": {"family": "jpeg", "params_version": "p1"},
                "attestation": {
                    "status": "absent",
                    "attestation_absent_reason": "attestation_disabled",
                    "final_event_attested_decision": {
                        "status": "absent",
                        "is_event_attested": False,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    observed_stage_names: list[str] = []

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        out_root = Path(cmd[cmd.index("--out") + 1])
        config_path = Path(cmd[cmd.index("--config") + 1])
        stage_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        stage_name = "calibration" if "run_calibrate" in " ".join(cmd) else "evaluate"
        observed_stage_names.append(stage_name)
        detect_records_glob = stage_cfg[stage_name]["detect_records_glob"]
        detect_records_path = Path(detect_records_glob)
        detect_records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(detect_records_path.parent.glob(detect_records_path.name))
        ]

        positive_payload = next(record for record in detect_records if record.get("label") is True)
        negative_payload = next(record for record in detect_records if record.get("label") is False)
        protocol_spec = {
            "family_field_candidates": ["attack_family", "attack.family", "attack.type"],
            "params_version_field_candidates": ["attack_params_version", "attack.params_version"],
        }

        assert positive_payload["attack_family"] == "jpeg"
        assert negative_payload["attack_family"] == "jpeg"
        assert metrics.build_attack_group_key(positive_payload, protocol_spec) == "jpeg::p1"
        assert metrics.build_attack_group_key(negative_payload, protocol_spec) == "jpeg::p1"
        assert positive_payload["attestation"]["attestation_source"] == "formal_input_payload"
        assert positive_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.89)
        assert negative_payload["attestation"]["attestation_source"] == "negative_branch_statement_only_provenance"
        assert negative_payload["attestation"]["authenticity_result"]["bundle_status"] == "statement_only_provenance_no_bundle"
        assert negative_payload["attestation"]["final_event_attested_decision"]["event_attestation_score"] == pytest.approx(0.0)

        (out_root / "records").mkdir(parents=True, exist_ok=True)
        (out_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
        if stage_name == "calibration":
            (out_root / "records" / "calibration_record.json").write_text(
                json.dumps(
                    {
                        "threshold_id": "event_attestation_score_np_fpr_0_01",
                        "threshold_key_used": "fpr_0_01",
                        "calibration_summary": {
                            "score_name": "event_attestation_score",
                            "threshold_value": 0.42,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_root / "artifacts" / "thresholds" / "thresholds_artifact.json").write_text(
                json.dumps(
                    {
                        "threshold_id": "event_attestation_score_np_fpr_0_01",
                        "score_name": "event_attestation_score",
                        "threshold_value": 0.42,
                        "threshold_key_used": "fpr_0_01",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            (out_root / "records" / "evaluate_record.json").write_text(
                json.dumps(
                    {
                        "threshold_key_used": "fpr_0_01",
                        "metrics": {
                            "score_name": "event_attestation_score",
                            "threshold_value": 0.42,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out_root / "artifacts" / "evaluation_report.json").write_text(
                json.dumps({"evaluation_report": {"score_name": "event_attestation_score"}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)

    module._run_parallel_attestation_statistics_workflow(
        repo_root=repo_root,
        run_root=run_root,
        cfg_path=cfg_path,
        profile="paper_full_cuda",
    )

    assert observed_stage_names == ["calibration", "evaluate"]


def test_onefile_workflow_runs_parallel_attestation_after_experiment_matrix_for_paper_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 paper profile 会在 experiment_matrix 成功后再执行并行 attestation 统计。

    Verify paper profile runs parallel attestation statistics only after the
    experiment_matrix step has completed successfully.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"evaluate": {}, "calibration": {}}, allow_unicode=True), encoding="utf-8")

    experiment_matrix_summary_path = run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json"
    step_order: list[str] = []

    monkeypatch.setattr(module, "_prepare_profile_cfg_path", lambda profile, run_root_arg, cfg_path_arg: cfg_path_arg)
    monkeypatch.setattr(module, "_prepare_experiment_matrix_cfg_path", lambda profile, run_root_arg, cfg_path_arg: cfg_path_arg)
    monkeypatch.setattr(module, "_prepare_stage_cfg_path", lambda *args, **kwargs: cfg_path)
    monkeypatch.setattr(
        module,
        "build_workflow_steps",
        lambda *args, **kwargs: [
            module.WorkflowStep(
                name="evaluate",
                command=[sys.executable, "-m", "main.cli.run_evaluate", "--out", str(run_root), "--config", str(cfg_path)],
                artifact_paths=[],
            ),
            module.WorkflowStep(
                name="experiment_matrix",
                command=[
                    sys.executable,
                    str(repo_root / "scripts" / "run_experiment_matrix.py"),
                    "--config",
                    str(cfg_path),
                    "--batch-root",
                    str(run_root / "outputs" / "experiment_matrix"),
                ],
                artifact_paths=[experiment_matrix_summary_path],
            ),
        ],
    )

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        command_text = " ".join(str(item) for item in cmd)
        if "main.cli.run_evaluate" in command_text:
            step_order.append("evaluate")
        elif "run_experiment_matrix.py" in command_text:
            step_order.append("experiment_matrix")
            experiment_matrix_summary_path.parent.mkdir(parents=True, exist_ok=True)
            experiment_matrix_summary_path.write_text(
                json.dumps({"total": 1, "failed": 0, "succeeded": 1}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)
    monkeypatch.setattr(module, "_run_parallel_attestation_statistics_workflow", lambda **kwargs: step_order.append("parallel"))
    monkeypatch.setattr(module, "_print_step_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_print_step_footer", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_print_artifact_presence", lambda *args, **kwargs: None)

    exit_code = module.run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile="paper_full_cuda",
        signoff_profile="paper",
        dry_run=False,
    )

    assert exit_code == 0
    assert step_order == ["evaluate", "experiment_matrix", "parallel"]


def test_dual_branch_negative_hf_only_sample_survives_formal_calibration_filters(tmp_path: Path) -> None:
    """
    功能：验证 dual-branch 负样本可从 detect_hf_score 恢复并通过 formal 校准过滤。 

    Verify dual-branch negative aggregation recovers HF-only score from
    detect_hf_score so calibration still has a valid null sample under formal
    exclusion rules.

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

    source_detect_path = records_dir / "detect_record.json"
    source_detect_path.write_text(
        json.dumps(
            {
                "label": True,
                "ground_truth": True,
                "is_watermarked": True,
                "content_evidence_payload": {
                    "status": "ok",
                    "score": 0.91,
                    "calibration_sample_usage": "formal_with_sidecar_disabled_marker",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    branch_neg_detect_path = tmp_path / "branch_neg_detect_record.json"
    branch_neg_detect_path.write_text(
        json.dumps(
            {
                "label": False,
                "ground_truth": False,
                "is_watermarked": False,
                "content_evidence_payload": {
                    "status": "absent",
                    "score": None,
                    "detect_lf_score": None,
                    "detect_hf_score": 0.37,
                    "content_failure_reason": "formal_profile_sidecar_disabled",
                },
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
        pair_count=1,
        branch_neg_detect_record=branch_neg_detect_path,
    )

    generated_positive = Path(glob_pattern.replace("*", "positive"))
    generated_negative = Path(glob_pattern.replace("*", "negative"))
    positive_payload = json.loads(generated_positive.read_text(encoding="utf-8"))
    negative_payload = json.loads(generated_negative.read_text(encoding="utf-8"))

    negative_content = negative_payload.get("content_evidence_payload")
    assert isinstance(negative_content, dict)
    assert negative_content.get("status") == "ok"
    assert negative_content.get("score") == 0.37
    assert negative_content.get("calibration_sample_usage") == "formal_with_dual_branch_negative_marker"

    scores, strata = load_scores_for_calibration(
        [positive_payload, negative_payload],
        cfg={
            "calibration": {
                "exclude_formal_sidecar_disabled_marker": True,
                "exclude_synthetic_negative_closure_marker": True,
            }
        },
    )
    assert scores == [0.37]
    sampling_policy = strata["sampling_policy"]
    assert sampling_policy["n_rejected_formal_sidecar_marker"] == 1
    assert sampling_policy["n_selected_null"] == 1


def test_prepare_detect_record_for_scoring_prefers_formal_hf_score_over_detect_hf_score(tmp_path: Path) -> None:
    """
    功能：验证 onefile 在存在正式 hf_score 时优先使用它，而不是 diagnostic detect_hf_score。

    Verify onefile prefers formal hf_score over diagnostic detect_hf_score when both exist.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    records_dir = run_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "detect_record.json").write_text(
        json.dumps(
            {
                "content_evidence_payload": {
                    "status": "absent",
                    "score": None,
                    "hf_score": 0.73,
                    "detect_hf_score": 0.37,
                    "content_failure_reason": "formal_profile_sidecar_disabled",
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = module._prepare_detect_record_for_scoring(
        run_root,
        records_dir,
        module.PROFILE_PAPER_FULL_CUDA,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    content_payload = payload["content_evidence_payload"]
    assert content_payload["score"] == 0.73


def test_onefile_attestation_hook_reuses_embed_record_payload(tmp_path: Path) -> None:
    """
    功能：验证 embed 后置 hook 只复用主路径 attestation 工件而不重算。

    Validate the embed post-hook materializes attestation artifacts directly from
    embed_record payload.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("attestation:\n  enabled: true\n", encoding="utf-8")
    (run_root / "records").mkdir(parents=True, exist_ok=True)

    embed_record = {
        "attestation": {
            "status": "ok",
            "statement": {"schema": "gen_attest_v1", "model_id": "m", "prompt_commit": "1", "seed_commit": "2", "plan_digest": "3", "event_nonce": "4", "time_bucket": "2026-03-13"},
            "attestation_digest": "aa" * 32,
            "event_binding_digest": "bb" * 32,
            "lf_payload_hex": "cc" * 16,
            "trace_commit": "dd" * 32,
            "geo_anchor_seed": 7,
            "signed_bundle": {"schema": "gen_attest_bundle_v1", "signature": {"signature_hex": "ee" * 64}},
        }
    }
    (run_root / "records" / "embed_record.json").write_text(
        json.dumps(embed_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    module._run_attestation_after_embed(run_root, cfg_path)

    statement_artifact = json.loads(
        (run_root / "artifacts" / "attestation" / "attestation_statement.json").read_text(encoding="utf-8")
    )
    bundle_artifact = json.loads(
        (run_root / "artifacts" / "attestation" / "attestation_bundle.json").read_text(encoding="utf-8")
    )

    assert statement_artifact["statement"] == embed_record["attestation"]["statement"]
    assert statement_artifact["event_binding_digest"] == "bb" * 32
    assert statement_artifact["trace_commit"] == "dd" * 32
    assert bundle_artifact == embed_record["attestation"]["signed_bundle"]


def test_onefile_attestation_verify_hook_reuses_detect_record_payload(tmp_path: Path) -> None:
    """
    功能：验证 detect 后置 hook 只复用主路径 attestation 结果而不重算。

    Validate the detect post-hook materializes attestation_result directly from
    detect_record payload.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "run_root"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("attestation:\n  enabled: true\n", encoding="utf-8")
    (run_root / "records").mkdir(parents=True, exist_ok=True)

    detect_record = {
        "attestation": {
            "verdict": "attested",
            "fusion_score": 0.92,
            "authenticity_result": {"status": "authentic"},
            "final_event_attested_decision": {"status": "attested", "is_event_attested": True},
        }
    }
    (run_root / "records" / "detect_record.json").write_text(
        json.dumps(detect_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    module._run_attestation_verification_after_detect(run_root, cfg_path)

    result_artifact = json.loads(
        (run_root / "artifacts" / "attestation" / "attestation_result.json").read_text(encoding="utf-8")
    )
    assert result_artifact == detect_record["attestation"]


def test_onefile_workflow_paper_full_profile_fails_fast_on_mismatched_impl(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda 对关键 impl 错配执行 fail-fast。 

    Verify paper_full_cuda rejects mismatched frozen impl bindings.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_full_bad_impl"
    cfg_obj = {
        "impl": {
            "sync_module_id": "geometry_latent_sync_sd3_v1",
            "geometry_extractor_id": "attention_anchor_extractor",
        }
    }
    cfg_path = tmp_path / "bad_impl_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="paper_full_cuda requires impl.sync_module_id"):
        module._prepare_profile_cfg_path("paper_full_cuda", run_root, cfg_path)


def test_onefile_workflow_paper_full_profile_keeps_unresolved_semantic_model_path(tmp_path: Path) -> None:
    """
    功能：验证 paper_full_cuda 在 semantic model 路径不可用时保留原路径并继续。

    Verify paper_full_cuda keeps unresolved semantic model path without startup hard-fail.

    Args:
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_full_missing_model"
    cfg_obj = {
        "impl": {
            "sync_module_id": "geometry_latent_sync_sd3",
            "geometry_extractor_id": "attention_anchor_extractor",
        },
        "mask": {
            "semantic_model_path": "/content/models/inspyrenet/inspyrenet_plus_ultra.pth",
        },
    }
    cfg_path = tmp_path / "missing_model.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False), encoding="utf-8")

    profile_cfg_path = module._prepare_profile_cfg_path("paper_full_cuda", run_root, cfg_path)
    profile_cfg_obj = yaml.safe_load(profile_cfg_path.read_text(encoding="utf-8"))
    assert profile_cfg_obj["mask"]["semantic_model_path"] == "/content/models/inspyrenet/inspyrenet_plus_ultra.pth"


def test_resolve_default_signoff_profile_for_profile() -> None:
    """
    功能：验证 signoff profile 默认解析策略。

    Validate default signoff profile resolution by workflow profile.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    assert module._resolve_default_signoff_profile_for_profile("cpu_smoke", None) == "baseline"
    assert module._resolve_default_signoff_profile_for_profile("paper_full_cuda", None) == "paper"
    assert module._resolve_default_signoff_profile_for_profile("paper_full_cuda", "publish") == "publish"


def test_build_stage_overrides_sets_embed_detect_content_switch() -> None:
    """
    功能：验证 embed/detect 阶段 content detect 开关显式写入 override。

    Validate explicit content detect switch overrides for embed and detect stages.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    embed_overrides = module._build_stage_overrides("embed", "paper_full_cuda")
    detect_overrides = module._build_stage_overrides("detect", "paper_full_cuda")

    assert "disable_content_detect=false" in embed_overrides
    assert "enable_content_detect=true" in detect_overrides


def test_run_redetect_promotes_canonical_detect_record_for_paper_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    功能：验证 paper profile 的 re-detect 会将 canonical detect record 回写主 run_root。

    Verify paper profile re-detect promotes the canonical detect record back to
    the main run_root detect record after calibrate.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary path fixture.

    Returns:
        None.
    """
    repo_root = Path(__file__).resolve().parent.parent
    module = _load_onefile_module(repo_root)

    run_root = tmp_path / "paper_full_run"
    (run_root / "artifacts" / "thresholds").mkdir(parents=True, exist_ok=True)
    (run_root / "records").mkdir(parents=True, exist_ok=True)

    thresholds_artifact = run_root / "artifacts" / "thresholds" / "thresholds_artifact.json"
    thresholds_artifact.write_text(
        json.dumps(
            {
                "threshold_id": "content_score_np_1e-02",
                "score_name": "content_score",
                "target_fpr": 0.01,
                "threshold_value": 1.11684145,
                "threshold_key_used": "content_score_np_1e-02",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_root / "records" / "embed_record.json").write_text(
        json.dumps({"status": "ok"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_root / "records" / "detect_record.json").write_text(
        json.dumps({"threshold_source": "fallback_target_fpr_test_only"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "paper_cfg.yaml"
    cfg_path.write_text("policy_path: content_np_geo_rescue\n", encoding="utf-8")

    controlled_copy_calls: list[dict[str, Any]] = []

    def _fake_run_subprocess(cmd: list[str], cwd: Path) -> int:
        _ = cwd
        detect_np_root = Path(cmd[cmd.index("--out") + 1])
        records_dir = detect_np_root / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        (records_dir / "detect_record.json").write_text(
            json.dumps(
                {
                    "threshold_source": "np_canonical",
                    "final_decision": {"threshold_source": "np_canonical"},
                    "fusion_result": {
                        "audit": {
                            "threshold_source": "np_canonical",
                            "allow_threshold_fallback_for_tests": False,
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    def _fake_controlled_copy(
        run_root: Path,
        artifacts_dir: Path,
        src_path: Path,
        dst_path: Path,
        kind: str = "artifact",
    ) -> None:
        controlled_copy_calls.append(
            {
                "run_root": run_root,
                "artifacts_dir": artifacts_dir,
                "src_path": src_path,
                "dst_path": dst_path,
                "kind": kind,
            }
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(module, "_run_subprocess_for_step", _fake_run_subprocess)
    monkeypatch.setattr(module.records_io, "copy_file_controlled_unbound", _fake_controlled_copy)

    module._run_redetect_with_np_thresholds(
        repo_root=repo_root,
        run_root=run_root,
        cfg_path=cfg_path,
        profile="paper_full_cuda",
    )

    assert len(controlled_copy_calls) == 1
    assert controlled_copy_calls[0]["run_root"] == run_root
    assert controlled_copy_calls[0]["artifacts_dir"] == run_root / "artifacts"
    assert controlled_copy_calls[0]["src_path"] == run_root / "artifacts" / "detect_np" / "records" / "detect_record.json"
    assert controlled_copy_calls[0]["dst_path"] == run_root / "records" / "detect_record.json"
    assert controlled_copy_calls[0]["kind"] == "record"

    promoted_record = json.loads((run_root / "records" / "detect_record.json").read_text(encoding="utf-8"))
    assert promoted_record.get("threshold_source") == "np_canonical"
    assert promoted_record.get("final_decision", {}).get("threshold_source") == "np_canonical"


