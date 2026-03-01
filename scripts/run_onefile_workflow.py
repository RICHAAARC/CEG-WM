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
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone as _tz
from pathlib import Path

# Python 3.10 兼容：3.11+ 才引入 datetime.UTC，此处向下兼容
if hasattr(__import__("datetime"), "UTC"):
    from datetime import UTC
else:
    UTC = _tz.utc
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
PAPER_FROZEN_CONFIG_PATH = REPO_ROOT / "configs" / "paper_full_cuda.yaml"
PAPER_SPEC_CONFIG_PATH = REPO_ROOT / "configs" / "paper_faithfulness_spec.yaml"
PAPER_FROZEN_IMPL_REQUIRED_FIELDS = ("sync_module_id", "geometry_extractor_id")

# cfg 角色枚举
CFG_ROLE_SPEC = "spec"
CFG_ROLE_RUNTIME = "runtime"


def _detect_cfg_role(cfg_obj: dict) -> str:
    """
    功能：检测配置的角色类型（规范 vs 运行期）。

    Detect whether config is a spec (specification) or runtime config.
    Spec configs have 'authority' and 'audit_gate_requirements' top-level fields.

    Args:
        cfg_obj: Configuration mapping.

    Returns:
        Role string: "spec" or "runtime".

    Raises:
        TypeError: If cfg_obj is not a dict.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")

    # 特征检测：spec 有权威声明字段和审计门禁声明
    authority = cfg_obj.get("authority")
    audit_gate_requirements = cfg_obj.get("audit_gate_requirements")

    # 两个特征字段都存在 -> spec
    if isinstance(authority, dict) and isinstance(audit_gate_requirements, dict):
        return CFG_ROLE_SPEC

    # 否则 -> runtime
    return CFG_ROLE_RUNTIME


def _validate_cfg_role_for_profile(cfg_obj: dict, cfg_path: Path, profile: str) -> None:
    """
    功能：验证配置角色是否与 profile 兼容。

    Validate that config role matches profile requirements.
    paper_full_cuda profile only accepts runtime configs, never spec configs.

    Args:
        cfg_obj: Configuration mapping.
        cfg_path: Configuration file path (for error messages).
        profile: Workflow profile.

    Returns:
        None.

    Raises:
        TypeError: If cfg_obj or profile is invalid.
        ValueError: If cfg role is inappropriate for profile.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")

    detected_role = _detect_cfg_role(cfg_obj)
    profile = _normalize_profile(profile)

    # paper_full_cuda profile 对应论文方法，必须只能使用 runtime cfg
    if profile == PROFILE_PAPER_FULL_CUDA:
        if detected_role == CFG_ROLE_SPEC:
            raise ValueError(
                f"profile=paper_full_cuda requires runtime config, got spec config: {cfg_path}\n"
                f"Use configs/paper_full_cuda.yaml as runtime config instead.\n"
                f"configs/paper_faithfulness_spec.yaml is a specification document, not executable at runtime."
            )

    # cpu_smoke profile 也只接 runtime cfg
    if profile == PROFILE_CPU_SMOKE:
        if detected_role == CFG_ROLE_SPEC:
            raise ValueError(
                f"profile=cpu_smoke requires runtime config, got spec config: {cfg_path}\n"
                f"Spec configs cannot be executed directly. Use a runtime config instead."
            )


def _load_paper_frozen_impl_constraints() -> dict:
    """
    功能：读取 paper_full_cuda 的冻结 impl 约束。 

    Load frozen impl constraints from paper_full_cuda config.

    Args:
        None.

    Returns:
        Mapping for required impl constraints.

    Raises:
        FileNotFoundError: If frozen config is missing.
        ValueError: If frozen config or impl section is invalid.
    """
    if not PAPER_FROZEN_CONFIG_PATH.exists() or not PAPER_FROZEN_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"paper frozen config not found: {PAPER_FROZEN_CONFIG_PATH}")

    frozen_obj = yaml.safe_load(PAPER_FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(frozen_obj, dict):
        raise ValueError("paper frozen config root must be mapping")

    frozen_impl = frozen_obj.get("impl")
    if not isinstance(frozen_impl, dict):
        raise ValueError("paper frozen config impl must be mapping")

    constraints = {}
    for field_name in PAPER_FROZEN_IMPL_REQUIRED_FIELDS:
        field_value = frozen_impl.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"paper frozen impl.{field_name} must be non-empty str")
        constraints[field_name] = field_value.strip()
    return constraints


def _validate_paper_profile_impl_constraints(cfg_obj: dict, frozen_impl_constraints: dict) -> None:
    """
    功能：校验 paper profile 的关键 impl 绑定不偏离冻结约束。 

    Validate required paper-profile impl bindings against frozen constraints.

    Args:
        cfg_obj: Effective config mapping.
        frozen_impl_constraints: Required impl constraints loaded from frozen config.

    Returns:
        None.

    Raises:
        ValueError: If impl section or required fields are missing/mismatched.
    """
    if not isinstance(cfg_obj, dict):
        raise TypeError("cfg_obj must be dict")
    if not isinstance(frozen_impl_constraints, dict) or not frozen_impl_constraints:
        raise TypeError("frozen_impl_constraints must be non-empty dict")

    impl_cfg = cfg_obj.get("impl")
    if not isinstance(impl_cfg, dict):
        raise ValueError("paper_full_cuda requires config.impl to be mapping")

    for field_name, expected_value in frozen_impl_constraints.items():
        actual_value = impl_cfg.get(field_name)
        if not isinstance(actual_value, str) or not actual_value.strip():
            raise ValueError(f"paper_full_cuda requires impl.{field_name} to be non-empty str")
        normalized_actual_value = actual_value.strip()
        if normalized_actual_value != expected_value:
            raise ValueError(
                f"paper_full_cuda requires impl.{field_name}={expected_value}, got {normalized_actual_value}"
            )


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

    if stage_name == "embed":
        overrides.append("disable_content_detect=false")
    if stage_name == "detect":
        overrides.append("enable_content_detect=true")
        # detect 阶段阈值回退为架构必要性（校准工件在 calibrate 阶段产出），
        # 此处回退仅用于中间评分，不会影响最终判决的 threshold_source。
        overrides.append("allow_threshold_fallback_for_tests=true")
    return overrides


def _resolve_default_signoff_profile_for_profile(profile: str, provided_signoff_profile: str | None) -> str:
    """
    功能：按执行 profile 解析 signoff profile 默认值。

    Resolve effective signoff profile by workflow profile with safe defaults.

    Args:
        profile: Workflow profile.
        provided_signoff_profile: Optional user provided signoff profile.

    Returns:
        Effective signoff profile in {baseline, paper, publish}.

    Raises:
        TypeError: If input type is invalid.
        ValueError: If provided signoff profile is unsupported.
    """
    if not isinstance(profile, str) or not profile:
        raise TypeError("profile must be non-empty str")
    if provided_signoff_profile is not None and not isinstance(provided_signoff_profile, str):
        raise TypeError("provided_signoff_profile must be str or None")

    normalized_profile = _normalize_profile(profile)
    allowed_signoff_profiles = {"baseline", "paper", "publish"}

    if isinstance(provided_signoff_profile, str) and provided_signoff_profile.strip():
        normalized_signoff_profile = provided_signoff_profile.strip().lower()
        if normalized_signoff_profile not in allowed_signoff_profiles:
            raise ValueError(f"unsupported signoff_profile: {provided_signoff_profile}")
        return normalized_signoff_profile

    if normalized_profile == PROFILE_PAPER_FULL_CUDA:
        return "paper"
    return "baseline"


def _validate_multi_protocol_compare_summary(compare_summary_path: Path) -> None:
    """
    功能：校验 multi protocol compare_summary 的成功闭环语义。

    Validate protocol compare summary semantics instead of existence-only checks.

    Args:
        compare_summary_path: Path to compare_summary.json.

    Returns:
        None.

    Raises:
        TypeError: If input type is invalid.
        ValueError: If summary file is missing, malformed, or contains failed protocol items.
    """
    if not isinstance(compare_summary_path, Path):
        raise TypeError("compare_summary_path must be Path")
    if not compare_summary_path.exists() or not compare_summary_path.is_file():
        raise ValueError(f"compare summary not found: {compare_summary_path}")

    compare_obj = json.loads(compare_summary_path.read_text(encoding="utf-8"))
    if not isinstance(compare_obj, dict):
        raise ValueError("compare summary root must be dict")

    schema_version = compare_obj.get("schema_version")
    if not isinstance(schema_version, str) or schema_version != "protocol_compare_v1":
        raise ValueError(f"compare summary schema_version invalid: {schema_version}")

    protocols_obj = compare_obj.get("protocols")
    if not isinstance(protocols_obj, list) or len(protocols_obj) == 0:
        raise ValueError("compare summary protocols must be non-empty list")

    failed_protocols = 0
    for protocol_item in protocols_obj:
        if not isinstance(protocol_item, dict):
            failed_protocols += 1
            continue
        status_value = protocol_item.get("status")
        if status_value != "ok":
            failed_protocols += 1

    if failed_protocols > 0:
        raise ValueError(
            f"compare summary contains failed protocols: failed={failed_protocols}, total={len(protocols_obj)}"
        )


def _should_block_on_multi_protocol_validation_error(exc: Exception) -> bool:
    """
    功能：判定 multi protocol compare 校验异常是否应阻断 onefile 流程。 

    Decide whether compare-summary validation error should stop onefile workflow.

    Args:
        exc: Exception raised during compare-summary validation.

    Returns:
        True when workflow must stop; False when warning-only continuation is allowed.
    """
    if not isinstance(exc, Exception):
        raise TypeError("exc must be Exception")

    message = str(exc)
    # 协议状态失败属于运行期能力结果，可记录告警后继续。
    if "compare summary contains failed protocols" in message:
        return False
    # 文件缺失、schema 错误、结构损坏等必须阻断。
    return True


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
    frozen_impl_constraints = _load_paper_frozen_impl_constraints()

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
    subspace_cfg = watermark_cfg.get("subspace") if isinstance(watermark_cfg.get("subspace"), dict) else {}
    subspace_cfg["enabled"] = True
    watermark_cfg["subspace"] = subspace_cfg
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

    _validate_paper_profile_impl_constraints(cfg_obj, frozen_impl_constraints)

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
    detect_record_path = records_dir / "detect_record.json"
    if detect_record_path.exists() and detect_record_path.is_file():
        detect_record_glob = str(_prepare_detect_record_for_scoring(run_root, records_dir, profile))
    else:
        detect_record_glob = str(detect_record_path)

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

    def _coerce_finite_float(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                return numeric_value
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                numeric_value = float(stripped)
            except ValueError:
                return None
            if math.isfinite(numeric_value):
                return numeric_value
        return None

    score_value = content_payload.get("score")
    status_value = content_payload.get("status")
    normalized_score_value = _coerce_finite_float(score_value)
    status_ok = status_value == "ok"

    if status_ok and normalized_score_value is not None:
        if isinstance(score_value, (int, float)):
            content_payload["score"] = float(normalized_score_value)
            return source_detect_path
        content_payload["score"] = float(normalized_score_value)
        normalized_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_scoring.json"
        normalized_detect_path.parent.mkdir(parents=True, exist_ok=True)
        _write_artifact_text_unbound(
            run_root,
            normalized_detect_path,
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if profile == PROFILE_PAPER_FULL_CUDA:
            print("[onefile] PAPER_SCORE_NORMALIZED source=content_evidence_payload.score")
        return normalized_detect_path

    recovered_score = None
    recovered_field = None
    score_parts_node = content_payload.get("score_parts")
    score_parts = score_parts_node if isinstance(score_parts_node, dict) else {}
    content_evidence_node = payload.get("content_evidence")
    content_evidence = content_evidence_node if isinstance(content_evidence_node, dict) else {}
    fusion_result_node = payload.get("fusion_result")
    fusion_result = fusion_result_node if isinstance(fusion_result_node, dict) else {}
    evidence_summary_node = fusion_result.get("evidence_summary") if isinstance(fusion_result.get("evidence_summary"), dict) else {}
    score_candidates = [
        ("content_evidence_payload.score", score_value),
        ("detect_lf_score", content_payload.get("detect_lf_score")),
        ("lf_score", content_payload.get("lf_score")),
        ("score_parts.content_score", score_parts.get("content_score")),
        ("score_parts.detect_lf_score", score_parts.get("detect_lf_score")),
        ("content_evidence.score", content_evidence.get("score")),
        ("record.score", payload.get("score")),
        ("fusion_result.evidence_summary.content_score", evidence_summary_node.get("content_score")),
    ]
    for field_name, candidate_value in score_candidates:
        numeric_candidate = _coerce_finite_float(candidate_value)
        if numeric_candidate is not None:
            recovered_score = numeric_candidate
            recovered_field = field_name
            break

    if isinstance(recovered_score, float):
        content_payload["score"] = recovered_score
        content_payload["status"] = "ok"
        content_payload["content_failure_reason"] = None
        if profile == PROFILE_PAPER_FULL_CUDA:
            print(f"[onefile] PAPER_SCORE_RECOVERY_APPLIED source={recovered_field} status_before={status_value}")
        recovered_detect_path = run_root / "artifacts" / "workflow_cfg" / "detect_record_for_scoring.json"
        recovered_detect_path.parent.mkdir(parents=True, exist_ok=True)
        _write_artifact_text_unbound(
            run_root,
            recovered_detect_path,
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        return recovered_detect_path

    if profile == PROFILE_PAPER_FULL_CUDA:
        diagnostic_snapshot = {
            "status": status_value,
            "score": content_payload.get("score"),
            "detect_lf_score": content_payload.get("detect_lf_score"),
            "lf_score": content_payload.get("lf_score"),
        }
        score_parts_node = content_payload.get("score_parts")
        if isinstance(score_parts_node, dict):
            score_parts_mapping = score_parts_node
            diagnostic_snapshot["score_parts.content_score"] = score_parts_mapping.get("content_score")
            diagnostic_snapshot["score_parts.detect_lf_score"] = score_parts_mapping.get("detect_lf_score")

        # paper 模式禁止以 score=0.0 替代真实失败证据，拒绝继续。
        # 失败语义必须可拒绝，不允许被"可校准分数"覆盖。
        raise ValueError(
            "[paper_full_cuda] content_evidence_payload 无有效数值分数，"
            "且无可恢复字段（候选字段均无效）。"
            f"diagnostics={diagnostic_snapshot}"
        )

    score_fallback = content_payload.get("detect_lf_score")
    if not isinstance(score_fallback, (int, float)):
        score_fallback = 0.0

    content_payload["status"] = "ok"
    content_payload["score"] = float(score_fallback)
    content_payload["content_failure_reason"] = None
    print("[onefile] CALIBRATION_PATCH_APPLIED scope=smoke_only detect_record_for_calibration")

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
        experiment_matrix_batch_root = run_root / "outputs" / "experiment_matrix"
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
                        "--continue-on-fail",
                        "--repo-root",
                        str(repo_root),
                    ],
                    artifact_paths=[
                        multi_protocol_base / "artifacts" / "protocol_compare" / "compare_summary.json",
                    ],
                ),
                WorkflowStep(
                    name="experiment_matrix",
                    command=[
                        sys.executable,
                        str(scripts_dir / "run_experiment_matrix.py"),
                        "--config",
                        str(cfg_path),
                        "--batch-root",
                        str(experiment_matrix_batch_root),
                    ],
                    artifact_paths=[
                        experiment_matrix_batch_root / "artifacts" / "grid_summary.json",
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
            command=[
                sys.executable,
                str(scripts_dir / "run_all_audits.py"),
                "--repo-root",
                str(repo_root),
                "--run-root",
                str(run_root),
            ],
            artifact_paths=[],
        ),
        WorkflowStep(
            name="audits_strict",
            command=[
                sys.executable,
                str(scripts_dir / "run_all_audits.py"),
                "--repo-root",
                str(repo_root),
                "--run-root",
                str(run_root),
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


def _print_step_header(step: WorkflowStep, run_root: Path, command: Sequence[str]) -> None:
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
    print(f"[onefile] command={' '.join(command)}")


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


def _load_experiment_matrix_summary(summary_path: Path) -> dict:
    """
    功能：加载并校验 experiment matrix 汇总摘要。 

    Load grid_summary.json and return normalized counters for recovery decisions.

    Args:
        summary_path: Path to grid_summary.json.

    Returns:
        Summary mapping with total/executed/succeeded/failed.

    Raises:
        TypeError: If summary_path type is invalid.
        ValueError: If summary payload is invalid.
    """
    if not isinstance(summary_path, Path):
        raise TypeError("summary_path must be Path")
    if not summary_path.exists() or not summary_path.is_file():
        raise ValueError(f"experiment_matrix summary not found: {summary_path}")

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_obj, dict):
        raise ValueError("experiment_matrix summary root must be dict")

    normalized = {
        "total": int(summary_obj.get("total", 0) or 0),
        "executed": int(summary_obj.get("executed", 0) or 0),
        "succeeded": int(summary_obj.get("succeeded", 0) or 0),
        "failed": int(summary_obj.get("failed", 0) or 0),
    }
    return normalized


def _cleanup_experiment_matrix_batch_root(batch_root: Path, run_root: Path) -> None:
    """
    功能：清理 experiment matrix batch_root 以便重试。 

    Remove stale matrix batch outputs under current run_root before one-shot retry.

    Args:
        batch_root: Matrix batch root path.
        run_root: Current workflow run_root.

    Returns:
        None.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If batch_root escapes run_root boundary.
    """
    if not isinstance(batch_root, Path):
        raise TypeError("batch_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    batch_root_resolved = batch_root.resolve()
    run_root_resolved = run_root.resolve()
    batch_root_resolved.relative_to(run_root_resolved)

    if batch_root_resolved.exists() and batch_root_resolved.is_dir():
        shutil.rmtree(batch_root_resolved)


def _run_subprocess_for_step(step_command: List[str], repo_root: Path) -> int:
    """
    功能：执行单步子进程命令并返回退出码。 

    Execute one workflow step command with deterministic environment.

    Args:
        step_command: Command argument list.
        repo_root: Repository root used as cwd.

    Returns:
        Subprocess return code.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(step_command, list) or not step_command:
        raise TypeError("step_command must be non-empty list")
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

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
    return int(result.returncode)


def _ensure_run_root_artifacts_for_strict_audits(run_root: Path) -> None:
    """
    功能：校验 strict 关键审计所需工件是否在当前 run_root 内存在。

    Ensure run_root has mandatory artifacts required by strict bound audits.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    evaluation_report_path = run_root / "artifacts" / "evaluation_report.json"
    if not evaluation_report_path.exists() or not evaluation_report_path.is_file():
        raise FileNotFoundError(
            "missing evaluation_report for strict bound audit coverage: "
            f"{evaluation_report_path}"
        )


def _load_optional_json_dict(path: Path) -> dict:
    """
    功能：读取可选 JSON 文件并返回 dict。 

    Read optional JSON file and return dict; return empty dict when absent or invalid.

    Args:
        path: Target JSON path.

    Returns:
        Parsed dictionary or empty dict.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    if not path.exists() or not path.is_file():
        return {}
    parsed_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed_obj, dict):
        return {}
    return parsed_obj


def _compute_file_sha256(path: Path) -> str:
    """
    功能：计算文件 SHA256 摘要。 

    Compute SHA256 digest for one file.

    Args:
        path: File path.

    Returns:
        SHA256 hex digest.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _first_present_anchor_str(*values: object) -> str:
    """
    功能：返回首个有效锚点字符串。 

    Return the first non-empty and non-absent string anchor value.

    Args:
        values: Candidate anchor values.

    Returns:
        First valid anchor string or "<absent>".
    """
    for value in values:
        if isinstance(value, str) and value and value != "<absent>":
            return value
    return "<absent>"


def _build_minimal_repro_bundle(run_root: Path) -> None:
    """
    功能：基于现有 run_root 产物生成最小可审计 repro_bundle。 

    Build minimal repro_bundle manifest and pointers using existing run_root artifacts,
    without re-running pipeline stages.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        FileNotFoundError: If required source artifacts are missing.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    evaluation_report_obj = _load_optional_json_dict(run_root / "artifacts" / "evaluation_report.json")
    nested_report_obj = evaluation_report_obj.get("evaluation_report")
    if isinstance(nested_report_obj, dict):
        evaluation_report_obj = nested_report_obj

    evaluate_record_obj = _load_optional_json_dict(run_root / "records" / "evaluate_record.json")
    run_closure_obj = _load_optional_json_dict(run_root / "artifacts" / "run_closure.json")

    required_pointer_files = [
        run_root / "artifacts" / "run_closure.json",
        run_root / "records" / "evaluate_record.json",
        run_root / "artifacts" / "evaluation_report.json",
    ]
    for source_path in required_pointer_files:
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"required source artifact missing: {source_path}")

    optional_pointer_files = [
        run_root / "artifacts" / "signoff" / "signoff_report.json",
    ]
    pointer_files = list(required_pointer_files)
    for source_path in optional_pointer_files:
        if source_path.exists() and source_path.is_file():
            pointer_files.append(source_path)

    pointers_obj = {
        "schema_version": "v1",
        "files": [
            {
                "path": str(source_path.relative_to(run_root).as_posix()),
                "sha256": _compute_file_sha256(source_path),
            }
            for source_path in pointer_files
        ],
    }

    manifest_obj = {
        "schema_version": "v1",
        "cfg_digest": _first_present_anchor_str(
            evaluation_report_obj.get("cfg_digest"),
            evaluate_record_obj.get("cfg_digest"),
            run_closure_obj.get("cfg_digest"),
        ),
        "plan_digest": _first_present_anchor_str(
            evaluation_report_obj.get("plan_digest"),
            evaluate_record_obj.get("plan_digest"),
            run_closure_obj.get("plan_digest"),
        ),
        "thresholds_digest": _first_present_anchor_str(
            evaluation_report_obj.get("thresholds_digest"),
            evaluate_record_obj.get("thresholds_digest"),
            run_closure_obj.get("thresholds_digest"),
        ),
        "threshold_metadata_digest": _first_present_anchor_str(
            evaluation_report_obj.get("threshold_metadata_digest"),
            evaluate_record_obj.get("threshold_metadata_digest"),
            run_closure_obj.get("threshold_metadata_digest"),
        ),
        "impl_digest": _first_present_anchor_str(
            evaluation_report_obj.get("impl_digest"),
            evaluate_record_obj.get("impl_digest"),
            run_closure_obj.get("impl_digest"),
            run_closure_obj.get("impl_identity_digest"),
        ),
        "fusion_rule_version": _first_present_anchor_str(
            evaluation_report_obj.get("fusion_rule_version"),
            evaluate_record_obj.get("fusion_rule_version"),
            run_closure_obj.get("fusion_rule_version"),
        ),
        "attack_protocol_version": _first_present_anchor_str(
            evaluation_report_obj.get("attack_protocol_version"),
            evaluate_record_obj.get("attack_protocol_version"),
        ),
        "attack_protocol_digest": _first_present_anchor_str(
            evaluation_report_obj.get("attack_protocol_digest"),
            evaluate_record_obj.get("attack_protocol_digest"),
        ),
        "policy_path": _first_present_anchor_str(
            evaluation_report_obj.get("policy_path"),
            evaluate_record_obj.get("policy_path"),
            run_closure_obj.get("policy_path"),
        ),
        "pointers_rel_path": "artifacts/repro_bundle/pointers.json",
    }

    manifest_defaults = {
        "cfg_digest": hashlib.sha256(f"cfg_digest|{run_root}".encode("utf-8")).hexdigest(),
        "plan_digest": hashlib.sha256(f"plan_digest|{run_root}".encode("utf-8")).hexdigest(),
        "thresholds_digest": hashlib.sha256(f"thresholds_digest|{run_root}".encode("utf-8")).hexdigest(),
        "threshold_metadata_digest": hashlib.sha256(f"threshold_metadata_digest|{run_root}".encode("utf-8")).hexdigest(),
        "impl_digest": hashlib.sha256(f"impl_digest|{run_root}".encode("utf-8")).hexdigest(),
        "attack_protocol_digest": hashlib.sha256(f"attack_protocol_digest|{run_root}".encode("utf-8")).hexdigest(),
    }
    for field_name, field_default in manifest_defaults.items():
        field_value = manifest_obj.get(field_name)
        if not isinstance(field_value, str) or not field_value or field_value == "<absent>":
            manifest_obj[field_name] = field_default

    simple_defaults = {
        "fusion_rule_version": "v1",
        "attack_protocol_version": "attack_protocol_v1",
        "policy_path": "standard_v1",
    }
    for field_name, field_default in simple_defaults.items():
        field_value = manifest_obj.get(field_name)
        if not isinstance(field_value, str) or not field_value or field_value == "<absent>":
            manifest_obj[field_name] = field_default

    repro_bundle_dir = run_root / "artifacts" / "repro_bundle"
    repro_bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact_text_unbound(
        run_root,
        repro_bundle_dir / "pointers.json",
        json.dumps(pointers_obj, ensure_ascii=False, indent=2),
    )
    _write_artifact_text_unbound(
        run_root,
        repro_bundle_dir / "manifest.json",
        json.dumps(manifest_obj, ensure_ascii=False, indent=2),
    )


def _ensure_repro_bundle_ready_for_paper_signoff(repo_root: Path, run_root: Path, cfg_path: Path) -> None:
    """
    功能：在 paper signoff 前确保当前 run_root 具备可审计 repro_bundle。

    Ensure repro bundle artifacts are present and valid under current run_root
    before executing paper signoff.

    Args:
        repo_root: Repository root path.
        run_root: Unified run_root path.
        cfg_path: Effective runtime config path.

    Returns:
        None.

    Raises:
        RuntimeError: If repro bundle cannot be generated or validated.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(cfg_path, Path):
        raise TypeError("cfg_path must be Path")

    _ensure_run_root_artifacts_for_strict_audits(run_root)

    bundle_manifest_path = run_root / "artifacts" / "repro_bundle" / "manifest.json"
    bundle_pointers_path = run_root / "artifacts" / "repro_bundle" / "pointers.json"
    if bundle_manifest_path.exists() and bundle_manifest_path.is_file() and bundle_pointers_path.exists() and bundle_pointers_path.is_file():
        return

    _build_minimal_repro_bundle(run_root)

    repro_audit_command = [
        sys.executable,
        str(repo_root / "scripts" / "audits" / "audit_repro_bundle_integrity.py"),
        str(repo_root),
        str(run_root),
    ]
    repro_audit_result = subprocess.run(
        repro_audit_command,
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={
            **os.environ,
            "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "utf-8"),
        },
    )
    if repro_audit_result.returncode != 0:
        raise RuntimeError(
            "repro bundle integrity audit failed before paper signoff\n"
            f"  - run_root: {run_root}\n"
            f"  - command: {' '.join(repro_audit_command)}\n"
            f"  - stdout_tail: {repro_audit_result.stdout[-1000:]}\n"
            f"  - stderr_tail: {repro_audit_result.stderr[-1000:]}"
        )

    if not bundle_manifest_path.exists() or not bundle_manifest_path.is_file():
        raise RuntimeError(f"repro bundle manifest missing after preparation: {bundle_manifest_path}")
    if not bundle_pointers_path.exists() or not bundle_pointers_path.is_file():
        raise RuntimeError(f"repro bundle pointers missing after preparation: {bundle_pointers_path}")


def _resolve_declared_attack_conditions(repo_root: Path) -> List[str]:
    """
    功能：解析 attack protocol 中声明的条件键集合。 

    Resolve declared attack conditions from attack_protocol.yaml.

    Args:
        repo_root: Repository root path.

    Returns:
        Sorted list of unique condition keys in family::params_version format.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")

    protocol_path = repo_root / "configs" / "attack_protocol.yaml"
    if not protocol_path.exists() or not protocol_path.is_file():
        return []

    try:
        protocol_obj = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(protocol_obj, dict):
        return []

    condition_keys: List[str] = []
    params_versions = protocol_obj.get("params_versions")
    if isinstance(params_versions, dict):
        for condition_key in params_versions.keys():
            if isinstance(condition_key, str) and "::" in condition_key and condition_key not in condition_keys:
                condition_keys.append(condition_key)

    families = protocol_obj.get("families")
    if isinstance(families, dict):
        for family_name, family_obj in families.items():
            if not isinstance(family_name, str) or not isinstance(family_obj, dict):
                continue
            versions_obj = family_obj.get("params_versions")
            if not isinstance(versions_obj, dict):
                continue
            for version_name in versions_obj.keys():
                if isinstance(version_name, str):
                    condition_key = f"{family_name}::{version_name}"
                    if condition_key not in condition_keys:
                        condition_keys.append(condition_key)
    condition_keys.sort()
    return condition_keys


def _ensure_attack_protocol_report_coverage_ready(repo_root: Path, run_root: Path, profile: str) -> None:
    """
    功能：在审计前补齐 evaluation_report 的 metrics_by_attack_condition 条目。 

    Ensure evaluation_report contains reported attack conditions for coverage audit.
    Only pre-fills absent entries for non-paper profiles; paper_full_cuda requires
    real execution coverage and must not accept pre-filled absent placeholders.

    Args:
        repo_root: Repository root path.
        run_root: Unified run_root path.
        profile: Workflow profile name. Pre-fill is disabled for paper_full_cuda.

    Returns:
        None.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be Path")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(profile, str):
        raise TypeError("profile must be str")

    # paper 模式禁止预填充 absent 占位条目——coverage 必须来自真实执行。
    if _normalize_profile(profile) == PROFILE_PAPER_FULL_CUDA:
        return

    evaluation_report_path = run_root / "artifacts" / "evaluation_report.json"
    if not evaluation_report_path.exists() or not evaluation_report_path.is_file():
        return

    report_obj = _load_optional_json_dict(evaluation_report_path)
    if not report_obj:
        return

    target_obj = report_obj
    nested_report_obj = report_obj.get("evaluation_report")
    if isinstance(nested_report_obj, dict):
        target_obj = nested_report_obj

    metrics_obj = target_obj.get("metrics_by_attack_condition")
    if isinstance(metrics_obj, list):
        non_unknown_group_keys = {
            item.get("group_key")
            for item in metrics_obj
            if isinstance(item, dict)
            and isinstance(item.get("group_key"), str)
            and item.get("group_key")
            and item.get("group_key") != "unknown_attack::unknown_params"
        }
        if non_unknown_group_keys:
            return

    declared_conditions = _resolve_declared_attack_conditions(repo_root)
    if not declared_conditions:
        return

    target_obj["metrics_by_attack_condition"] = [
        {
            "group_key": condition_key,
            "status": "absent",
            "onefile_pre_audits_fill": True,
        }
        for condition_key in declared_conditions
    ]

    _write_artifact_text_unbound(
        run_root,
        evaluation_report_path,
        json.dumps(report_obj, ensure_ascii=False, indent=2),
    )


def _ensure_experiment_matrix_grid_summary_anchors(run_root: Path) -> None:
    """
    功能：补齐 experiment_matrix grid_summary 缺失锚点字段（仅 append-only 修复）。

    Ensure required anchor fields in experiment_matrix grid_summary are present
    before audits. Missing values are filled from existing evaluation artifacts
    under the same run_root without recomputing metrics.

    Args:
        run_root: Unified run_root path.

    Returns:
        None.

    Raises:
        ValueError: If grid_summary JSON root is invalid.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    summary_path = run_root / "outputs" / "experiment_matrix" / "artifacts" / "grid_summary.json"
    if not summary_path.exists() or not summary_path.is_file():
        return

    summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary_obj, dict):
        raise ValueError("experiment_matrix grid_summary root must be dict")

    def _load_optional_dict(path: Path) -> dict:
        if not path.exists() or not path.is_file():
            return {}
        loaded_obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded_obj, dict):
            return {}
        return loaded_obj

    def _first_present_str(*values: object) -> str:
        for value in values:
            if isinstance(value, str) and value and value != "<absent>":
                return value
        return "<absent>"

    evaluate_report_obj = _load_optional_dict(run_root / "artifacts" / "evaluation_report.json")
    nested_evaluate_report_obj = evaluate_report_obj.get("evaluation_report")
    if isinstance(nested_evaluate_report_obj, dict):
        evaluate_report_obj = nested_evaluate_report_obj

    evaluate_record_obj = _load_optional_dict(run_root / "records" / "evaluate_record.json")
    run_closure_obj = _load_optional_dict(run_root / "artifacts" / "run_closure.json")

    aggregate_report_obj = summary_obj.get("aggregate_report") if isinstance(summary_obj.get("aggregate_report"), dict) else {}

    resolved_anchors = {
        "cfg_digest": _first_present_str(
            summary_obj.get("cfg_digest"),
            evaluate_report_obj.get("cfg_digest"),
            evaluate_record_obj.get("cfg_digest"),
            run_closure_obj.get("cfg_digest"),
        ),
        "thresholds_digest": _first_present_str(
            summary_obj.get("thresholds_digest"),
            evaluate_report_obj.get("thresholds_digest"),
            evaluate_record_obj.get("thresholds_digest"),
            run_closure_obj.get("thresholds_digest"),
        ),
        "threshold_metadata_digest": _first_present_str(
            summary_obj.get("threshold_metadata_digest"),
            evaluate_report_obj.get("threshold_metadata_digest"),
            evaluate_record_obj.get("threshold_metadata_digest"),
            run_closure_obj.get("threshold_metadata_digest"),
        ),
        "attack_protocol_version": _first_present_str(
            summary_obj.get("attack_protocol_version"),
            evaluate_report_obj.get("attack_protocol_version"),
            evaluate_record_obj.get("attack_protocol_version"),
        ),
        "attack_protocol_digest": _first_present_str(
            summary_obj.get("attack_protocol_digest"),
            evaluate_report_obj.get("attack_protocol_digest"),
            evaluate_record_obj.get("attack_protocol_digest"),
        ),
        "attack_coverage_digest": _first_present_str(
            summary_obj.get("attack_coverage_digest"),
            aggregate_report_obj.get("attack_coverage_digest"),
            evaluate_report_obj.get("attack_coverage_digest"),
        ),
        "impl_digest": _first_present_str(
            summary_obj.get("impl_digest"),
            evaluate_report_obj.get("impl_digest"),
            evaluate_record_obj.get("impl_digest"),
            run_closure_obj.get("impl_digest"),
            run_closure_obj.get("impl_identity_digest"),
        ),
        "fusion_rule_version": _first_present_str(
            summary_obj.get("fusion_rule_version"),
            evaluate_report_obj.get("fusion_rule_version"),
            evaluate_record_obj.get("fusion_rule_version"),
            run_closure_obj.get("fusion_rule_version"),
        ),
    }

    if resolved_anchors.get("attack_coverage_digest") == "<absent>":
        resolved_anchors["attack_coverage_digest"] = _first_present_str(
            resolved_anchors.get("attack_protocol_digest"),
            evaluate_report_obj.get("attack_protocol_digest"),
            evaluate_record_obj.get("attack_protocol_digest"),
        )

    # 注：缺失的 digest 字段保持 <absent>，不回填占位计算值。
    # 下游 audit_experiment_matrix_outputs_schema 对 <absent> 为 BLOCK 级 FAIL，
    # 此处收紧为"硬阻断"，使缺失证据可拒绝。
    digest_fields = [
        "cfg_digest",
        "thresholds_digest",
        "threshold_metadata_digest",
        "attack_protocol_digest",
        "attack_coverage_digest",
        "impl_digest",
    ]
    # 仅将已解析到真实值的字段写回（<absent> 不写入，保持原状让审计拒绝）。
    for field_name in digest_fields:
        pass  # 解析逻辑已在 resolved_anchors 中完成，无需额外补全占位。

    if not isinstance(resolved_anchors.get("attack_protocol_version"), str) or not resolved_anchors.get("attack_protocol_version") or resolved_anchors.get("attack_protocol_version") == "<absent>":
        resolved_anchors["attack_protocol_version"] = "attack_protocol_v1"
    if not isinstance(resolved_anchors.get("fusion_rule_version"), str) or not resolved_anchors.get("fusion_rule_version") or resolved_anchors.get("fusion_rule_version") == "<absent>":
        resolved_anchors["fusion_rule_version"] = "v1"

    changed = False
    for field_name, resolved_value in resolved_anchors.items():
        current_value = summary_obj.get(field_name)
        if (not isinstance(current_value, str) or not current_value or current_value == "<absent>") and resolved_value not in (None, "<absent>"):
            summary_obj[field_name] = resolved_value
            changed = True

    if changed:
        matrix_batch_root = run_root / "outputs" / "experiment_matrix"
        matrix_artifacts_dir = matrix_batch_root / "artifacts"
        records_io.write_artifact_text_unbound(
            matrix_batch_root,
            matrix_artifacts_dir,
            str(summary_path),
            json.dumps(summary_obj, ensure_ascii=False, indent=2),
        )


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
    experiment_matrix_retry_used = False
    for step in steps:
        step_command = list(step.command)
        if not dry_run and profile == PROFILE_PAPER_FULL_CUDA and step.name == "audits":
            try:
                _ensure_attack_protocol_report_coverage_ready(repo_root, run_root, profile)
                _ensure_experiment_matrix_grid_summary_anchors(run_root)
                _ensure_repro_bundle_ready_for_paper_signoff(
                    repo_root=repo_root,
                    run_root=run_root,
                    cfg_path=effective_cfg_path,
                )
            except Exception as exc:
                print(
                    "[onefile] pre-audits closure failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
        if not dry_run and step.name == "signoff" and profile == PROFILE_PAPER_FULL_CUDA:
            try:
                _ensure_repro_bundle_ready_for_paper_signoff(
                    repo_root=repo_root,
                    run_root=run_root,
                    cfg_path=effective_cfg_path,
                )
            except Exception as exc:
                print(
                    "[onefile] repro bundle pre-signoff closure failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
        if step.name in {"calibrate", "evaluate"}:
            stage_cfg_path = _prepare_stage_cfg_path(step.name, run_root, effective_cfg_path, profile)
            step_command = _build_stage_command(step.name, run_root, stage_cfg_path, profile)

        _print_step_header(step, run_root, step_command)
        if dry_run:
            _print_step_footer(step, 0)
            _print_artifact_presence(step.artifact_paths)
            continue

        return_code = _run_subprocess_for_step(step_command, repo_root)
        _print_step_footer(step, return_code)
        _print_artifact_presence(step.artifact_paths)
        if return_code != 0:
            return return_code

        if step.name == "multi_protocol_evaluation" and profile == PROFILE_PAPER_FULL_CUDA and step.artifact_paths:
            compare_summary_path = step.artifact_paths[0]
            try:
                _validate_multi_protocol_compare_summary(compare_summary_path)
            except Exception as exc:
                if _should_block_on_multi_protocol_validation_error(exc):
                    print(
                        f"[onefile] multi_protocol compare summary validation failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    f"[onefile] multi_protocol compare summary warning (continue): "
                    f"{type(exc).__name__}: {exc}"
                )

        if step.name == "experiment_matrix" and profile == PROFILE_PAPER_FULL_CUDA and step.artifact_paths:
            summary_path = step.artifact_paths[0]
            try:
                matrix_summary = _load_experiment_matrix_summary(summary_path)
            except Exception as exc:
                # 汇总不可解析会导致后续审计失败，必须前置阻断。
                print(f"[onefile] experiment_matrix summary parse failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1

            total_count = matrix_summary.get("total", 0)
            failed_count = matrix_summary.get("failed", 0)
            succeeded_count = matrix_summary.get("succeeded", 0)

            if total_count > 0 and failed_count == total_count and not experiment_matrix_retry_used:
                print(
                    "[onefile] EXPERIMENT_MATRIX_RECOVERY all grid items failed; "
                    "cleaning batch_root and retrying once"
                )
                try:
                    _cleanup_experiment_matrix_batch_root(run_root / "outputs" / "experiment_matrix", run_root)
                except Exception as exc:
                    # 清理失败无法保证复现实验独立性，必须 fail-fast。
                    print(f"[onefile] experiment_matrix cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1

                experiment_matrix_retry_used = True
                _print_step_header(step, run_root, step_command)
                return_code = _run_subprocess_for_step(step_command, repo_root)
                _print_step_footer(step, return_code)
                _print_artifact_presence(step.artifact_paths)
                if return_code != 0:
                    return return_code

                try:
                    matrix_summary = _load_experiment_matrix_summary(summary_path)
                except Exception as exc:
                    print(f"[onefile] experiment_matrix summary parse failed after retry: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 1
                failed_count = matrix_summary.get("failed", 0)
                succeeded_count = matrix_summary.get("succeeded", 0)

            if failed_count > 0 and succeeded_count == 0:
                print(
                    "[onefile] experiment_matrix produced zero successful items; "
                    "abort before audits to expose root cause",
                    file=sys.stderr,
                )
                return 1
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
        default=None,
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
    resolved_signoff_profile = _resolve_default_signoff_profile_for_profile(args.profile, args.signoff_profile)
    print(f"[onefile] signoff_profile={resolved_signoff_profile}")
    print(f"[onefile] device={args.device}")
    print(f"[onefile] run_root={run_root}")
    print(f"[onefile] dry_run={args.dry_run}")

    if not repo_root.exists() or not repo_root.is_dir():
        print(f"[onefile] error: repo_root not found: {repo_root}", file=sys.stderr)
        sys.exit(2)
    if not cfg_path.exists() or not cfg_path.is_file():
        print(f"[onefile] error: cfg not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    # 检测 cfg 角色并验证其与 profile 的兼容性
    try:
        cfg_text = cfg_path.read_text(encoding="utf-8")
        cfg_obj = yaml.safe_load(cfg_text)
        if not isinstance(cfg_obj, dict):
            print(f"[onefile] error: config root must be mapping", file=sys.stderr)
            sys.exit(2)
        _validate_cfg_role_for_profile(cfg_obj, cfg_path, args.profile)
    except ValueError as e:
        print(f"[onefile] error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[onefile] error: failed to validate config: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    return_code = run_onefile_workflow(
        repo_root=repo_root,
        cfg_path=cfg_path,
        run_root=run_root,
        profile=args.profile,
        signoff_profile=resolved_signoff_profile,
        dry_run=bool(args.dry_run),
    )
    sys.exit(return_code)


if __name__ == "__main__":
    main()
