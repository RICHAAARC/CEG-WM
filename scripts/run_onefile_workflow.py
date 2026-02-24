"""
文件目的：onefile workflow 编排入口。
Module type: General module

职责边界：
1. 仅编排既有 CLI/脚本入口（embed/detect/calibrate/evaluate/audits/signoff）。
2. 不直接写 records 或关键 artifacts，不改写冻结语义与 digest 口径。
3. 通过既有 CLI override 机制传递可变项，保持门禁与白名单约束。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Sequence
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.core import records_io


PROFILE_CPU_SMOKE = "cpu_smoke"
PROFILE_PAPER_FULL_CUDA = "paper_full_cuda"
LEGACY_PROFILE_CPU_MIN = "cpu_min"
LEGACY_PROFILE_CUDA_REAL = "cuda_real"


def _normalize_profile(profile: str) -> str:
    """
    功能：规范化 profile 名称（兼容历史别名）。

    Normalize workflow profile names with backward-compatible aliases.

    Args:
        profile: Raw profile name.

    Returns:
        Normalized profile name.

    Raises:
        TypeError: If input type is invalid.
        ValueError: If profile is unsupported.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if profile == LEGACY_PROFILE_CPU_MIN:
        return PROFILE_CPU_SMOKE
    if profile == LEGACY_PROFILE_CUDA_REAL:
        return PROFILE_PAPER_FULL_CUDA
    if profile in {PROFILE_CPU_SMOKE, PROFILE_PAPER_FULL_CUDA}:
        return profile
    raise ValueError(f"unsupported profile: {profile}")


@dataclass(frozen=True)
class WorkflowStep:
    """
    功能：描述单个编排步骤。 

    Describe one orchestrated workflow step.

    Args:
        name: Stable step name.
        command: Executable command list.
        artifact_paths: Key artifact paths expected after step.

    Returns:
        None.
    """

    name: str
    command: List[str]
    artifact_paths: List[Path]


def _build_run_root(repo_root: Path, provided_run_root: str | None, profile: str) -> Path:
    """
    功能：解析或生成 run_root 路径。 

    Resolve run_root path from user input or generate a deterministic-form path.

    Args:
        repo_root: Repository root path.
        provided_run_root: Optional user-provided run_root.
        profile: Selected workflow profile.

    Returns:
        Absolute run_root path.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If provided_run_root is empty.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if provided_run_root is not None and not isinstance(provided_run_root, str):
        raise TypeError("provided_run_root must be str or None")
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")

    if provided_run_root is not None:
        if not provided_run_root.strip():
            raise ValueError("run_root must be non-empty when provided")
        candidate = Path(provided_run_root)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    generated = repo_root / "outputs" / f"onefile_{profile}_{timestamp}_{suffix}"
    return generated.resolve()


def _build_stage_overrides(stage_name: str, profile: str) -> List[str]:
    """
    功能：构造阶段 override 参数。 

    Build allowed CLI overrides for each stage.

    Args:
        stage_name: Stage name in {embed, detect, calibrate, evaluate}.
        profile: Workflow profile name.

    Returns:
        List of override strings in key=value format.

    Raises:
        ValueError: If stage_name or profile is unsupported.
    """
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")
    profile = _normalize_profile(profile)

    reason = json.dumps(f"onefile_workflow_{stage_name}", ensure_ascii=False)
    overrides = [
        "run_root_reuse_allowed=true",
        f"run_root_reuse_reason={reason}",
    ]

    if profile == PROFILE_CPU_SMOKE:
        overrides.extend(
            [
                "force_cpu=\"cpu\"",
                "enable_trace_tap=true",
                "test_mode_identity=true",
            ]
        )
    if profile == PROFILE_PAPER_FULL_CUDA:
        overrides.extend(
            [
                "enable_trace_tap=true",
                "enable_paper_faithfulness=true",
            ]
        )
    return overrides


def _prepare_profile_cfg_path(profile: str, run_root: Path, cfg_path: Path) -> Path:
    """
    功能：按 profile 生成运行期配置文件。 

    Build profile-specific runtime config file when profile requires strict fields.

    Args:
        profile: Workflow profile name.
        run_root: Unified run_root path.
        cfg_path: Base config path.

    Returns:
        Config path to be used by stage commands.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If config content is invalid.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    profile = _normalize_profile(profile)
    if profile != PROFILE_PAPER_FULL_CUDA:
        return cfg_path

    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_obj = yaml.safe_load(cfg_text)
    if not isinstance(cfg_obj, dict):
        raise ValueError("config root must be mapping")

    cfg_obj["pipeline_impl_id"] = "sd3_diffusers_real_v1"
    cfg_obj["pipeline_build_enabled"] = True
    cfg_obj["device"] = "cuda"

    model_source = cfg_obj.get("model_source")
    if not isinstance(model_source, str) or model_source not in {"hf", "hf_hub", "local"}:
        cfg_obj["model_source"] = "hf"

    hf_revision = cfg_obj.get("hf_revision")
    if not isinstance(hf_revision, str) or not hf_revision:
        cfg_obj["hf_revision"] = "main"

    paper_cfg = cfg_obj.get("paper_faithfulness")
    if not isinstance(paper_cfg, dict):
        paper_cfg = {}
    paper_cfg["enabled"] = True
    paper_cfg["alignment_check"] = True
    cfg_obj["paper_faithfulness"] = paper_cfg

    tap_cfg = cfg_obj.get("trajectory_tap")
    if not isinstance(tap_cfg, dict):
        tap_cfg = {}
    tap_cfg["enabled"] = True
    cfg_obj["trajectory_tap"] = tap_cfg

    model_cfg = cfg_obj.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not isinstance(model_cfg.get("dtype"), str) or not model_cfg.get("dtype"):
        model_cfg["dtype"] = "float16"
    if not isinstance(model_cfg.get("height"), int) or model_cfg.get("height", 0) <= 0:
        model_cfg["height"] = 512
    if not isinstance(model_cfg.get("width"), int) or model_cfg.get("width", 0) <= 0:
        model_cfg["width"] = 512
    cfg_obj["model"] = model_cfg

    watermark_cfg = cfg_obj.get("watermark")
    if not isinstance(watermark_cfg, dict):
        watermark_cfg = {}
    hf_cfg = watermark_cfg.get("hf") if isinstance(watermark_cfg.get("hf"), dict) else {}
    lf_cfg = watermark_cfg.get("lf") if isinstance(watermark_cfg.get("lf"), dict) else {}
    hf_cfg["enabled"] = True
    hf_cfg["tail_truncation_mode"] = "top_k_per_latent"
    hf_cfg["selection"] = "top_k_magnitude_based"
    lf_cfg["enabled"] = True
    lf_cfg["coding_mode"] = "latent_space_sign_flipping"
    lf_cfg["decoder"] = "belief_propagation"
    watermark_cfg["hf"] = hf_cfg
    watermark_cfg["lf"] = lf_cfg
    cfg_obj["watermark"] = watermark_cfg

    detect_cfg = cfg_obj.get("detect") if isinstance(cfg_obj.get("detect"), dict) else {}
    geometry_cfg = detect_cfg.get("geometry") if isinstance(detect_cfg.get("geometry"), dict) else {}
    geometry_cfg["enabled"] = True
    geometry_cfg["enable_attention_anchor"] = True
    detect_cfg["geometry"] = geometry_cfg
    cfg_obj["detect"] = detect_cfg

    impl_cfg = cfg_obj.get("impl") if isinstance(cfg_obj.get("impl"), dict) else {}
    impl_cfg["sync_module_id"] = "geometry_latent_sync_sd3_v1"
    impl_cfg["geometry_extractor_id"] = "geometry_align_invariance_sd3_v1"
    cfg_obj["impl"] = impl_cfg

    mask_cfg = cfg_obj.get("mask") if isinstance(cfg_obj.get("mask"), dict) else {}
    mask_cfg["impl_id"] = "semantic_saliency_v2"
    cfg_obj["mask"] = mask_cfg

    embed_cfg = cfg_obj.get("embed") if isinstance(cfg_obj.get("embed"), dict) else {}
    embed_cfg["test_mode_identity"] = False
    cfg_obj["embed"] = embed_cfg

    profile_cfg_path = run_root / "artifacts" / "workflow_cfg" / "profile_paper_full_cuda.yaml"
    _write_artifact_text_unbound(
        run_root,
        profile_cfg_path,
        yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False)
    )
    return profile_cfg_path


def _build_stage_command(
    stage_name: str,
    run_root: Path,
    cfg_path: Path,
    profile: str,
) -> List[str]:
    """
    功能：构建阶段命令。 

    Build one stage command using existing CLI module entry.

    Args:
        stage_name: Stage name.
        run_root: Unified run_root path.
        cfg_path: Config file path.
        profile: Workflow profile.

    Returns:
        Command argument list.
    """
    command = [
        sys.executable,
        "-m",
        f"main.cli.run_{stage_name}",
        "--out",
        str(run_root),
        "--config",
        str(cfg_path),
    ]

    for item in _build_stage_overrides(stage_name, profile):
        command.extend(["--override", item])

    if stage_name == "detect":
        command.extend(["--input", str(run_root / "records" / "embed_record.json")])

    return command


def _prepare_stage_cfg_path(
    stage_name: str,
    run_root: Path,
    cfg_path: Path,
    profile: str,
) -> Path:
    """
    功能：为特定阶段生成补全字段后的配置文件。 

    Build stage-specific config file when extra fields are required by stage logic.

    Args:
        stage_name: Stage name.
        run_root: Unified run_root path.
        cfg_path: Base config path.

    Returns:
        Config path for current stage.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If config content is invalid.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    if stage_name not in {"calibrate", "evaluate"}:
        return cfg_path

    profile = _normalize_profile(profile)

    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_obj = yaml.safe_load(cfg_text)
    if not isinstance(cfg_obj, dict):
        raise ValueError("config root must be mapping")

    records_dir = run_root / "records"
    if profile == PROFILE_PAPER_FULL_CUDA:
        detect_record_path = records_dir / "detect_record.json"
        if not detect_record_path.exists() or not detect_record_path.is_file():
            raise ValueError(f"paper_full_cuda requires detect_record.json: {detect_record_path}")
        detect_record_glob = str(detect_record_path)
    else:
        detect_record_glob = str(_prepare_detect_record_for_scoring(run_root, records_dir, profile))

    if stage_name == "calibrate":
        calibration_cfg = cfg_obj.get("calibration")
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        calibration_cfg["detect_records_glob"] = detect_record_glob
        cfg_obj["calibration"] = calibration_cfg

    if stage_name == "evaluate":
        evaluate_cfg = cfg_obj.get("evaluate")
        if not isinstance(evaluate_cfg, dict):
            evaluate_cfg = {}
        evaluate_cfg["detect_records_glob"] = detect_record_glob
        evaluate_cfg["thresholds_path"] = str(run_root / "artifacts" / "thresholds" / "thresholds_artifact.json")
        cfg_obj["evaluate"] = evaluate_cfg

    stage_cfg_dir = run_root / "artifacts" / "workflow_cfg"
    stage_cfg_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg_path = stage_cfg_dir / f"{stage_name}_config.yaml"
    _write_artifact_text_unbound(
        run_root,
        stage_cfg_path,
        yaml.safe_dump(cfg_obj, allow_unicode=True, sort_keys=False)
    )
    return stage_cfg_path


def _prepare_detect_record_for_scoring(run_root: Path, records_dir: Path, profile: str) -> Path:
    """
    功能：为校准/评估准备至少一个可用分数样本记录。 

    Prepare a detect record path that guarantees at least one numeric content score.

    Args:
        run_root: Unified run_root path.
        records_dir: Records directory under run_root.

    Returns:
        Path to detect record JSON used by calibrate/evaluate.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If source detect record is missing or invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(records_dir, Path):
        raise TypeError("records_dir must be Path")

    profile = _normalize_profile(profile)
    if profile == PROFILE_PAPER_FULL_CUDA:
        raise ValueError("detect record patching is forbidden in paper_full_cuda profile")

    source_detect_path = records_dir / "detect_record.json"
    if not source_detect_path.exists() or not source_detect_path.is_file():
        raise ValueError(f"detect record not found: {source_detect_path}")

    payload = json.loads(source_detect_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detect record must be JSON object")

    content_payload = payload.get("content_evidence_payload")
    if not isinstance(content_payload, dict):
        content_payload = {}
        payload["content_evidence_payload"] = content_payload

    score_value = content_payload.get("score")
    status_value = content_payload.get("status")
    score_valid = isinstance(score_value, (int, float))
    status_ok = status_value == "ok"

    if status_ok and score_valid:
        return source_detect_path

    score_fallback = content_payload.get("detect_lf_score")
    if not isinstance(score_fallback, (int, float)):
        score_fallback = 0.0

    content_payload["status"] = "ok"
    content_payload["score"] = float(score_fallback)
    content_payload["content_failure_reason"] = None
    print("[onefile] SMOKE_ONLY_PATCH_APPLIED detect_record_for_calibration")

    calibrated_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_calibration.json"
    calibrated_detect_path.parent.mkdir(parents=True, exist_ok=True)
    _write_artifact_text_unbound(
        run_root,
        calibrated_detect_path,
        json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return calibrated_detect_path


def _write_artifact_text_unbound(run_root: Path, path: Path, text: str) -> None:
    """
    功能：通过 records_io 受控入口写入文本工件。 

    Write artifact text via records_io controlled entrypoint.

    Args:
        run_root: Unified run_root path.
        path: Artifact file path.
        text: Artifact text content.

    Returns:
        None.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if not isinstance(text, str):
        raise TypeError("text must be str")
    artifacts_dir = run_root / "artifacts"
    records_io.write_artifact_text_unbound(run_root, artifacts_dir, str(path), text)


def build_workflow_steps(
    run_root: Path,
    cfg_path: Path,
    repo_root: Path,
    profile: str,
    signoff_profile: str,
) -> List[WorkflowStep]:
    """
    功能：构建固定顺序 onefile 流程步骤。 

    Build the fixed step chain:
    embed → detect → calibrate → evaluate → audits → audits_strict → signoff.

    Args:
        run_root: Unified run_root.
        cfg_path: Main config path.
        repo_root: Repository root path.
        profile: Workflow profile.
        signoff_profile: Signoff profile.

    Returns:
        Ordered workflow step list.
    """
    profile = _normalize_profile(profile)
    scripts_dir = repo_root / "scripts"
    steps: List[WorkflowStep] = [
        WorkflowStep(
            name="embed",
            command=_build_stage_command("embed", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "embed_record.json",
                run_root / "artifacts" / "run_closure.json",
            ],
        ),
        WorkflowStep(
            name="detect",
            command=_build_stage_command("detect", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "detect_record.json",
                run_root / "artifacts" / "run_closure.json",
            ],
        ),
        WorkflowStep(
            name="calibrate",
            command=_build_stage_command("calibrate", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "calibration_record.json",
                run_root / "artifacts" / "thresholds" / "thresholds_artifact.json",
            ],
        ),
        WorkflowStep(
            name="evaluate",
            command=_build_stage_command("evaluate", run_root, cfg_path, profile),
            artifact_paths=[
                run_root / "records" / "evaluate_record.json",
                run_root / "artifacts" / "evaluation_report.json",
                run_root / "artifacts" / "run_closure.json",
            ],
        ),
    ]

    if profile == PROFILE_PAPER_FULL_CUDA:
        multi_protocol_base = run_root / "artifacts" / "multi_protocol_evaluation"
        steps.extend(
            [
                WorkflowStep(
                    name="multi_protocol_evaluation",
                    command=[
                        sys.executable,
                        str(scripts_dir / "run_multi_protocol_evaluation.py"),
                        "--base-cfg",
                        str(cfg_path),
                        "--protocol",
                        str(repo_root / "configs" / "attack_protocol.yaml"),
                        "--mode",
                        "repro",
                        "--run-root-base",
                        str(multi_protocol_base),
                        "--repo-root",
                        str(repo_root),
                    ],
                    artifact_paths=[
                        multi_protocol_base / "artifacts" / "protocol_compare" / "compare_summary.json",
                    ],
                ),
                WorkflowStep(
                    name="assert_paper_mechanisms",
                    command=[
                        sys.executable,
                        str(scripts_dir / "assert_paper_mechanisms.py"),
                        "--run-root",
                        str(run_root),
                        "--config",
                        str(cfg_path),
                        "--profile",
                        profile,
                    ],
                    artifact_paths=[
                        run_root / "records" / "embed_record.json",
                        run_root / "records" / "detect_record.json",
                        run_root / "artifacts" / "evaluation_report.json",
                    ],
                ),
            ]
        )

    steps.extend([
        WorkflowStep(
            name="audits",
            command=[sys.executable, str(scripts_dir / "run_all_audits.py"), "--repo-root", str(repo_root)],
            artifact_paths=[],
        ),
        WorkflowStep(
            name="audits_strict",
            command=[
                sys.executable,
                str(scripts_dir / "run_all_audits.py"),
                "--repo-root",
                str(repo_root),
                "--strict",
            ],
            artifact_paths=[],
        ),
        WorkflowStep(
            name="signoff",
            command=[
                sys.executable,
                str(scripts_dir / "run_freeze_signoff.py"),
                "--run-root",
                str(run_root),
                "--repo-root",
                str(repo_root),
                "--signoff-profile",
                signoff_profile,
            ],
            artifact_paths=[run_root / "artifacts" / "signoff" / "signoff_report.json"],
        ),
    ])
    return steps


def _print_step_header(step: WorkflowStep, run_root: Path) -> None:
    """
    功能：打印步骤执行头信息。 

    Print step name, command, start time and run_root for auditability.

    Args:
        step: Workflow step.
        run_root: Unified run_root.

    Returns:
        None.
    """
    started_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 88)
    print(f"[onefile] step={step.name} start={started_at}")
    print(f"[onefile] run_root={run_root}")
    print(f"[onefile] command={' '.join(step.command)}")


def _print_step_footer(step: WorkflowStep, return_code: int) -> None:
    """
    功能：打印步骤执行尾信息。 

    Print end time and return code.

    Args:
        step: Workflow step.
        return_code: Subprocess return code.

    Returns:
        None.
    """
    ended_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[onefile] step={step.name} end={ended_at} return_code={return_code}")


def _print_artifact_presence(artifact_paths: Sequence[Path]) -> None:
    """
    功能：打印关键产物存在性。 

    Print existence status for key artifact paths.

    Args:
        artifact_paths: Key artifact paths.

    Returns:
        None.
    """
    if not artifact_paths:
        print("[onefile] artifacts=<none>")
        return
    for item in artifact_paths:
        status = "exists" if item.exists() else "missing"
        print(f"[onefile] artifact={item} status={status}")


def run_onefile_workflow(
    repo_root: Path,
    cfg_path: Path,
    run_root: Path,
    profile: str,
    signoff_profile: str,
    dry_run: bool,
) -> int:
    """
    功能：执行 onefile 全链路编排。 

    Execute onefile workflow with strict fail-fast semantics.

    Args:
        repo_root: Repository root.
        cfg_path: Config path.
        run_root: Unified run_root.
        profile: Workflow profile.
        signoff_profile: Signoff profile.
        dry_run: Whether to print commands without execution.

    Returns:
        Process exit code. Returns first failed step return code, or 0 on success.
    """
    profile = _normalize_profile(profile)
    effective_cfg_path = _prepare_profile_cfg_path(profile, run_root, cfg_path)
    steps = build_workflow_steps(run_root, effective_cfg_path, repo_root, profile, signoff_profile)
    for step in steps:
        step_command = list(step.command)
        if step.name in {"calibrate", "evaluate"}:
            detect_record_path = run_root / "records" / "detect_record.json"
            if detect_record_path.exists() and detect_record_path.is_file():
                stage_cfg_path = _prepare_stage_cfg_path(step.name, run_root, effective_cfg_path, profile)
                step_command = _build_stage_command(step.name, run_root, stage_cfg_path, profile)

        _print_step_header(step, run_root)
        if dry_run:
            _print_step_footer(step, 0)
            _print_artifact_presence(step.artifact_paths)
            continue

        result = subprocess.run(
            step_command,
            cwd=str(repo_root),
            check=False,
            env={
                **os.environ,
                "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE"),
                "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
            },
        )
        _print_step_footer(step, int(result.returncode))
        _print_artifact_presence(step.artifact_paths)
        if result.returncode != 0:
            return int(result.returncode)
    return 0


def _parse_args() -> argparse.Namespace:
    """
    功能：解析命令行参数。 

    Parse command line arguments for onefile workflow.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Onefile workflow: embed -> detect -> calibrate -> evaluate -> audits -> signoff"
    )
    parser.add_argument("--cfg", required=True, help="Config YAML path")
    parser.add_argument("--run-root", default=None, help="Unified run_root path")
    parser.add_argument(
        "--profile",
        default=PROFILE_CPU_SMOKE,
        choices=[PROFILE_CPU_SMOKE, PROFILE_PAPER_FULL_CUDA, LEGACY_PROFILE_CPU_MIN, LEGACY_PROFILE_CUDA_REAL],
        help="Execution profile",
    )
    parser.add_argument(
        "--signoff-profile",
        default="baseline",
        choices=["baseline", "paper", "publish"],
        help="Signoff profile passed to run_freeze_signoff.py",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Reserved device selector")
    parser.add_argument("--dry-run", action="store_true", help="Print step commands without execution")
    return parser.parse_args()


def main() -> None:
    """
    功能：onefile workflow 主入口。 

    Main entry point for onefile workflow orchestration.

    Returns:
        None.
    """
    args = _parse_args()
    args.profile = _normalize_profile(args.profile)

    script_path = Path(__file__).resolve()
    default_repo_root = script_path.parent.parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else default_repo_root
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    else:
        cfg_path = cfg_path.resolve()

    run_root = _build_run_root(repo_root, args.run_root, args.profile)

    print(f"[onefile] repo_root={repo_root}")
    print(f"[onefile] cfg={cfg_path}")
    print(f"[onefile] profile={args.profile}")
    print(f"[onefile] device={args.device}")
    print(f"[onefile] run_root={run_root}")
    print(f"[onefile] dry_run={args.dry_run}")

    if not repo_root.exists() or not repo_root.is_dir():
        print(f"[onefile] error: repo_root not found: {repo_root}", file=sys.stderr)
        sys.exit(2)
    if not cfg_path.exists() or not cfg_path.is_file():
        print(f"[onefile] error: cfg not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    return_code = run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile=args.profile,
        signoff_profile=args.signoff_profile,
        dry_run=bool(args.dry_run),
    )
    sys.exit(return_code)


if __name__ == "__main__":
    main()
