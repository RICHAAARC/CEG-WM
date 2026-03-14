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
                    "content_failure_reason": "image_domain_sidecar_disabled_by_ablation",
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


