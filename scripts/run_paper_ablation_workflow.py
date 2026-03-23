#!/usr/bin/env python3
"""
文件目的：paper 单对象单变量消融编排脚本。
Module type: General module

职责边界：
1. 仅复用既有 embed/detect/calibrate/evaluate CLI，不改写方法实现。
2. 面向 Colab / notebook 场景，执行单次 embed、多次 detect 的同对象消融。
3. 负责输出目录组织、variant 配置快照与 compare summary 生成。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import yaml

_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from main.core import config_loader


WORKFLOW_SECTION_KEY = "paper_ablation_workflow"
NOTEBOOK_CONFIG_TEMPLATE_VERSION = "paper_ablation_workflow_v2"

DEFAULT_COMPARE_SUMMARY_FIELDS = [
    "variant_name",
    "variant_suffix",
    "variant_group",
    "variant_category",
    "attestation_status",
    "attestation_result_status",
    "content_attestation_score",
    "event_attestation_score",
    "channel_scores_lf",
    "channel_scores_hf",
    "channel_scores_geo",
    "active_score_source",
    "active_geo_score_source",
    "geo_repair_enabled",
    "geo_repair_active",
    "geo_repair_mode",
    "geo_score_repair_enabled",
    "geo_score_repair_active",
    "geo_score_repair_mode",
    "geo_repair_direction_classification",
    "lf_exact_repair_enabled",
    "lf_exact_repair_applied",
    "lf_exact_repair_mode",
    "formal_exact_evidence_source",
    "protocol_root_cause_classification",
    "variant_run_root",
    "base_embed_record_path",
    "detect_record_path",
    "input_record_path",
    "config_snapshot_path",
    "variant_overrides",
]

DEFAULT_COMPARE_TABLE_FIELDS = [
    "variant_name",
    "variant_suffix",
    "variant_group",
    "variant_category",
    "attestation_status",
    "content_attestation_score",
    "event_attestation_score",
    "channel_scores_lf",
    "channel_scores_hf",
    "channel_scores_geo",
    "active_geo_score_source",
    "geo_repair_enabled",
    "geo_repair_active",
    "geo_repair_mode",
    "geo_score_repair_enabled",
    "geo_score_repair_active",
    "geo_score_repair_mode",
    "geo_repair_direction_classification",
    "lf_exact_repair_enabled",
    "lf_exact_repair_applied",
    "lf_exact_repair_mode",
    "formal_exact_evidence_source",
    "protocol_root_cause_classification",
    "base_embed_record_path",
    "detect_record_path",
]

RUNTIME_RESOURCE_DOTTED_PATHS = [
    "mask.semantic_model_path",
    "inference_prompt_file",
    "embed.input_image_path",
]

_MISSING = object()


@dataclass(frozen=True)
class AblationVariant:
    """
    功能：表示单个消融 variant 定义。

    Structured single-variant definition for paper ablation orchestration.

    Args:
        name: Human-readable variant name.
        suffix: Stable directory suffix for outputs.
        description: Variant description.
        enabled: Whether the variant is enabled by default.
        group: Optional single-variable comparison group label.
        category: Optional variant category label.
        overrides: Dotted-path overrides applied to the detect-side config snapshot.
        override_group_name: Optional explicit group label when a variant needs
            more than one coordinated override.

    Returns:
        None.
    """

    name: str
    suffix: str
    description: str
    enabled: bool
    group: str | None
    category: str | None
    overrides: Dict[str, Any]
    override_group_name: str | None = None


@dataclass(frozen=True)
class AblationLayout:
    """
    功能：表示 ablation 工作流的输出目录布局。

    Structured output layout for one ablation run.

    Args:
        run_root: Top-level ablation output root.
        base_embed_root: Base embed run root.
        variants_root: Directory containing per-variant detect run roots.
        compare_root: Directory containing compare artifacts.
        config_snapshot_root: Directory for generated config snapshots.

    Returns:
        None.
    """

    run_root: Path
    base_embed_root: Path
    variants_root: Path
    compare_root: Path
    config_snapshot_root: Path


def _resolve_repo_path(path_value: Any) -> Path:
    """
    功能：将仓库内相对路径解析为绝对路径。

    Resolve a repository-relative path into an absolute path.

    Args:
        path_value: Relative or absolute path value.

    Returns:
        Resolved absolute path.

    Raises:
        TypeError: If path_value is invalid.
    """
    if isinstance(path_value, Path):
        candidate = path_value
    elif isinstance(path_value, str) and path_value.strip():
        candidate = Path(path_value.strip())
    else:
        raise TypeError("path_value must be non-empty str or Path")
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root / candidate).resolve()


def _read_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    功能：读取 YAML 配置并要求根对象为 dict。

    Load a YAML config file and require a mapping root.

    Args:
        config_path: YAML config path.

    Returns:
        Loaded configuration mapping.

    Raises:
        FileNotFoundError: If config_path does not exist.
        TypeError: If the loaded root object is not a mapping.
    """
    cfg_obj, _ = config_loader.load_yaml_with_provenance(config_path)
    if not isinstance(cfg_obj, dict):
        raise TypeError("config root must be dict")
    return dict(cfg_obj)


def _require_mapping(node: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(node, dict):
        raise TypeError(f"{field_name} must be dict")
    return cast(Dict[str, Any], dict(node))


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be non-empty str")
    return value.strip()


def _require_optional_non_empty_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_str(value, field_name)


def _require_string_list(value: Any, field_name: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{field_name} must be non-empty list[str]")
    return [_require_non_empty_str(item, field_name) for item in cast(List[Any], value)]


def _normalize_dotted_overrides(overrides: Any, field_name: str) -> Dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise TypeError(f"{field_name} must be dict")
    normalized: Dict[str, Any] = {}
    override_items = cast(Dict[Any, Any], overrides)
    for dotted_key, dotted_value in override_items.items():
        normalized_key = _require_non_empty_str(dotted_key, f"{field_name}.key")
        normalized[normalized_key] = dotted_value
    return normalized


def _load_variant_definitions(workflow_cfg: Dict[str, Any]) -> List[AblationVariant]:
    """
    功能：解析并验证 workflow 中的 variant 列表。

    Parse and validate variant definitions from the workflow config.

    Args:
        workflow_cfg: paper_ablation_workflow config mapping.

    Returns:
        Ordered list of parsed variant definitions.

    Raises:
        TypeError: If variant structure is invalid.
        ValueError: If variant naming or override constraints are violated.
    """
    detect_rerun_cfg = _require_mapping(workflow_cfg.get("detect_rerun"), "paper_ablation_workflow.detect_rerun")
    strict_single_variable = bool(detect_rerun_cfg.get("strict_single_variable", True))
    variants_node = detect_rerun_cfg.get("variants")
    if not isinstance(variants_node, list) or not variants_node:
        raise TypeError("paper_ablation_workflow.detect_rerun.variants must be non-empty list")
    variant_list = cast(List[Any], variants_node)

    variants: List[AblationVariant] = []
    seen_names: set[str] = set()
    seen_suffixes: set[str] = set()
    for index, variant_node in enumerate(variant_list):
        if not isinstance(variant_node, dict):
            raise TypeError(f"variant[{index}] must be dict")
        variant_cfg = cast(Dict[str, Any], dict(variant_node))
        name = _require_non_empty_str(variant_cfg.get("name"), f"variant[{index}].name")
        suffix = _require_non_empty_str(variant_cfg.get("suffix", name), f"variant[{index}].suffix")
        description = _require_non_empty_str(
            variant_cfg.get("description", f"paper ablation variant {name}"),
            f"variant[{index}].description",
        )
        enabled = bool(variant_cfg.get("enabled", True))
        group = _require_optional_non_empty_str(variant_cfg.get("group"), f"variant[{index}].group")
        category = _require_optional_non_empty_str(variant_cfg.get("category"), f"variant[{index}].category")
        overrides = _normalize_dotted_overrides(variant_cfg.get("overrides"), f"variant[{index}].overrides")
        override_group_name = variant_cfg.get("override_group_name")
        if override_group_name is not None:
            override_group_name = _require_non_empty_str(
                override_group_name,
                f"variant[{index}].override_group_name",
            )

        if name in seen_names:
            raise ValueError(f"duplicate variant name: {name}")
        if suffix in seen_suffixes:
            raise ValueError(f"duplicate variant suffix: {suffix}")
        if strict_single_variable and len(overrides) > 1 and override_group_name is None:
            raise ValueError(
                "strict_single_variable variants must contain at most one override unless "
                "override_group_name is provided: "
                f"variant={name} override_count={len(overrides)}"
            )

        seen_names.add(name)
        seen_suffixes.add(suffix)
        variants.append(
            AblationVariant(
                name=name,
                suffix=suffix,
                description=description,
                enabled=enabled,
                group=group,
                category=category,
                overrides=overrides,
                override_group_name=override_group_name,
            )
        )
    return variants


def _resolve_workflow_cfg(cfg_obj: Any) -> Dict[str, Any]:
    """
    功能：提取并验证 paper ablation workflow 配置段。

    Resolve and validate the dedicated paper ablation workflow section.

    Args:
        cfg_obj: Runtime configuration mapping.

    Returns:
        Normalized paper_ablation_workflow mapping.

    Raises:
        TypeError: If required workflow nodes are invalid.
    """
    workflow_node = cfg_obj.get(WORKFLOW_SECTION_KEY)
    workflow_cfg = _require_mapping(workflow_node, WORKFLOW_SECTION_KEY)
    _require_mapping(workflow_cfg.get("base_embed"), f"{WORKFLOW_SECTION_KEY}.base_embed")
    _require_mapping(workflow_cfg.get("detect_rerun"), f"{WORKFLOW_SECTION_KEY}.detect_rerun")
    _require_mapping(workflow_cfg.get("compare"), f"{WORKFLOW_SECTION_KEY}.compare")
    _require_mapping(workflow_cfg.get("notebook_runtime"), f"{WORKFLOW_SECTION_KEY}.notebook_runtime")
    _load_variant_definitions(workflow_cfg)
    return workflow_cfg


def _get_dotted_value(node: Any, dotted_path: str) -> Any:
    current = node
    for key_name in dotted_path.split("."):
        if not isinstance(current, dict) or key_name not in current:
            return _MISSING
        current = current[key_name]
    return current


def _pick_first_present_value(node: Dict[str, Any], dotted_paths: List[str]) -> Any:
    for dotted_path in dotted_paths:
        candidate = _get_dotted_value(node, dotted_path)
        if candidate is not _MISSING:
            return candidate
    return None


def _resolve_compare_fields(workflow_cfg: Dict[str, Any], field_name: str, default_fields: List[str]) -> List[str]:
    compare_cfg = _require_mapping(workflow_cfg.get("compare"), f"{WORKFLOW_SECTION_KEY}.compare")
    configured_fields = compare_cfg.get(field_name)
    if configured_fields is None:
        return list(default_fields)
    return _require_string_list(configured_fields, f"{WORKFLOW_SECTION_KEY}.compare.{field_name}")


def _resolve_notebook_runtime_cfg(workflow_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _require_mapping(workflow_cfg.get("notebook_runtime"), f"{WORKFLOW_SECTION_KEY}.notebook_runtime")


def _apply_nested_override(cfg_obj: Dict[str, Any], dotted_path: str, value: Any) -> None:
    current = cfg_obj
    path_parts = dotted_path.split(".")
    for key_name in path_parts[:-1]:
        existing = current.get(key_name)
        if existing is None:
            current[key_name] = {}
            existing = current[key_name]
        if not isinstance(existing, dict):
            raise TypeError(f"override target is not mapping: {dotted_path}")
        current = existing
    current[path_parts[-1]] = value


def _resolve_runtime_resource_path(configured_path: str, source_config_path: Path) -> str:
    """
    功能：将运行时配置中的资源路径解析为稳定的绝对路径。

    Resolve runtime-config resource paths into stable absolute paths before
    stage-specific snapshots are written under outputs/.

    Args:
        configured_path: Path string from the runtime config.
        source_config_path: Runtime config path that provided the resource field.

    Returns:
        Absolute path string when a stable existing candidate is found;
        otherwise the original configured path.

    Raises:
        TypeError: If input types are invalid.
    """
    if not configured_path.strip():
        raise TypeError("configured_path must be non-empty str")

    normalized_path = configured_path.strip()
    candidate_path = Path(normalized_path).expanduser()
    if candidate_path.is_absolute():
        resolved_candidate = candidate_path.resolve()
        if resolved_candidate.exists():
            return str(resolved_candidate)
        return normalized_path

    search_candidates: List[Path] = []
    cfg_relative_candidate = (source_config_path.parent / candidate_path).resolve()
    search_candidates.append(cfg_relative_candidate)

    repo_relative_candidate = (_repo_root / candidate_path).resolve()
    if repo_relative_candidate not in search_candidates:
        search_candidates.append(repo_relative_candidate)

    sanitized_parts = [part for part in candidate_path.parts if part not in {".", ".."}]
    if sanitized_parts:
        repo_sanitized_candidate = _repo_root.joinpath(*sanitized_parts).resolve()
        if repo_sanitized_candidate not in search_candidates:
            search_candidates.append(repo_sanitized_candidate)

    for resolved_candidate in search_candidates:
        if resolved_candidate.exists():
            return str(resolved_candidate)
    return normalized_path


def _normalize_runtime_resource_paths(cfg_snapshot: Dict[str, Any], source_config_path: Path) -> Dict[str, Any]:
    """
    功能：修正 notebook runtime config 写入 outputs 后失效的资源路径。

    Normalize known repository asset paths after notebook runtime configs are
    materialized outside configs/ and before downstream CLI snapshots are used.

    Args:
        cfg_snapshot: Snapshot mapping without workflow-only fields.
        source_config_path: Runtime config path used to build the snapshot.

    Returns:
        Snapshot mapping with normalized resource paths where resolvable.

    Raises:
        TypeError: If inputs are invalid.
    """
    for dotted_path in RUNTIME_RESOURCE_DOTTED_PATHS:
        current_value = _get_dotted_value(cfg_snapshot, dotted_path)
        if not isinstance(current_value, str) or not current_value.strip():
            continue
        normalized_value = _resolve_runtime_resource_path(current_value, source_config_path)
        if normalized_value != current_value:
            _apply_nested_override(cfg_snapshot, dotted_path, normalized_value)
    return cfg_snapshot


def _build_runtime_cfg_snapshot(
    base_cfg_obj: Dict[str, Any],
    overrides: Dict[str, Any],
    source_config_path: Path,
) -> Dict[str, Any]:
    snapshot = cast(Dict[str, Any], json.loads(json.dumps(base_cfg_obj, ensure_ascii=False)))
    snapshot.pop(WORKFLOW_SECTION_KEY, None)
    for dotted_path, override_value in overrides.items():
        _apply_nested_override(snapshot, dotted_path, override_value)
    _normalize_runtime_resource_paths(snapshot, source_config_path)
    return snapshot


def _write_yaml_file(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _ensure_clean_run_root(
    run_root: Path,
    *,
    fresh_run: bool,
    resume: bool,
    reuse_base_embed_record: Path | None,
) -> None:
    if fresh_run and resume:
        raise ValueError("fresh_run and resume cannot both be true")
    if fresh_run and reuse_base_embed_record is not None:
        raise ValueError("fresh_run cannot be combined with reuse_base_embed_record")
    if run_root.exists():
        if not run_root.is_dir():
            raise ValueError(f"run_root exists and is not a directory: {run_root}")
        has_entries = any(run_root.iterdir())
        if fresh_run and has_entries:
            raise ValueError(f"run_root must be empty for fresh_run mode: {run_root}")
        if has_entries and not resume and reuse_base_embed_record is None:
            raise ValueError(
                "run_root is non-empty; use --fresh-run to require an empty run root or "
                "--resume to continue an existing run"
            )
    run_root.mkdir(parents=True, exist_ok=True)


def _resolve_layout(workflow_cfg: Dict[str, Any], run_root: Path) -> AblationLayout:
    base_embed_cfg = _require_mapping(workflow_cfg.get("base_embed"), f"{WORKFLOW_SECTION_KEY}.base_embed")
    detect_rerun_cfg = _require_mapping(workflow_cfg.get("detect_rerun"), f"{WORKFLOW_SECTION_KEY}.detect_rerun")
    config_snapshot_dir = _require_non_empty_str(
        workflow_cfg.get("config_snapshot_dir", "compare/config_snapshots"),
        f"{WORKFLOW_SECTION_KEY}.config_snapshot_dir",
    )
    base_embed_root = run_root / _require_non_empty_str(
        base_embed_cfg.get("run_subdir", "base_embed"),
        f"{WORKFLOW_SECTION_KEY}.base_embed.run_subdir",
    )
    variants_root = run_root / _require_non_empty_str(
        detect_rerun_cfg.get("variants_dir", "variants"),
        f"{WORKFLOW_SECTION_KEY}.detect_rerun.variants_dir",
    )
    compare_root = run_root / _require_non_empty_str(
        detect_rerun_cfg.get("compare_dir", "compare"),
        f"{WORKFLOW_SECTION_KEY}.detect_rerun.compare_dir",
    )
    config_snapshot_root = run_root / config_snapshot_dir
    base_embed_root.mkdir(parents=True, exist_ok=True)
    variants_root.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)
    config_snapshot_root.mkdir(parents=True, exist_ok=True)
    return AblationLayout(
        run_root=run_root,
        base_embed_root=base_embed_root,
        variants_root=variants_root,
        compare_root=compare_root,
        config_snapshot_root=config_snapshot_root,
    )


def _run_subprocess(command: List[str], cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print("[paper_ablation][dry-run] " + " ".join(command))
        return
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "subprocess failed: "
            f"returncode={result.returncode}; command={' '.join(command)}; "
            f"stdout={result.stdout}; stderr={result.stderr}"
        )


def _build_stage_overrides(stage_name: str) -> List[str]:
    """
    功能：构造 paper ablation workflow 的阶段级 CLI overrides。

    Build stage-specific CLI overrides aligned with the validated onefile paper
    workflow conventions.

    Args:
        stage_name: Stage name in {embed, detect, calibrate, evaluate}.

    Returns:
        Ordered CLI override strings.

    Raises:
        ValueError: If stage_name is unsupported.
    """
    if stage_name not in {"embed", "detect", "calibrate", "evaluate"}:
        raise ValueError(f"unsupported stage_name: {stage_name}")

    reason = json.dumps(f"paper_ablation_workflow_{stage_name}", ensure_ascii=False)
    overrides = [
        "run_root_reuse_allowed=true",
        f"run_root_reuse_reason={reason}",
        "enable_trace_tap=true",
        "enable_paper_faithfulness=true",
    ]
    if stage_name == "embed":
        overrides.append("disable_content_detect=false")
    elif stage_name == "detect":
        overrides.append("enable_content_detect=true")
        overrides.append("allow_threshold_fallback_for_tests=true")
    return overrides


def _build_embed_command(run_root: Path, config_path: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "main.cli.run_embed",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]
    for item in _build_stage_overrides("embed"):
        command.extend(["--override", item])
    return command


def _build_detect_command(run_root: Path, config_path: Path, input_record_path: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "main.cli.run_detect",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
        "--input",
        str(input_record_path),
    ]
    for item in _build_stage_overrides("detect"):
        command.extend(["--override", item])
    return command


def _build_calibrate_command(run_root: Path, config_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "main.cli.run_calibrate",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]


def _build_evaluate_command(run_root: Path, config_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "main.cli.run_evaluate",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be dict: {path}")
    return cast(Dict[str, Any], payload)


def _as_dict(value: Any) -> Dict[str, Any]:
    return cast(Dict[str, Any], dict(value)) if isinstance(value, dict) else {}


def _extract_variant_compare_row(
    variant: AblationVariant,
    variant_run_root: Path,
    base_embed_record_path: Path,
    detect_record_path: Path,
    input_record_path: Path,
    config_snapshot_path: Path,
    variant_cfg_snapshot: Dict[str, Any],
    detect_record: Dict[str, Any],
) -> Dict[str, Any]:
    attestation_payload = _as_dict(detect_record.get("attestation"))
    image_evidence_result = _as_dict(attestation_payload.get("image_evidence_result"))
    final_event_decision = _as_dict(attestation_payload.get("final_event_attested_decision"))
    content_payload = _as_dict(detect_record.get("content_evidence_payload"))
    geometry_payload = _as_dict(detect_record.get("geometry_evidence_payload"))
    geo_rescue_diagnostics = _as_dict(
        _pick_first_present_value(
            detect_record,
            [
                "geo_rescue_diagnostics_artifact",
                "_geo_rescue_diagnostics_artifact",
                "artifacts.geo_rescue_diagnostics_artifact",
            ],
        )
    )
    score_parts = _as_dict(content_payload.get("score_parts"))
    channel_scores = _as_dict(attestation_payload.get("channel_scores"))
    image_channel_scores = _as_dict(image_evidence_result.get("channel_scores"))
    variant_detect_cfg = _as_dict(variant_cfg_snapshot.get("detect"))
    variant_geometry_cfg = _as_dict(variant_detect_cfg.get("geometry"))
    variant_geo_repair_cfg = _as_dict(variant_geometry_cfg.get("geo_score_repair"))
    variant_content_cfg = _as_dict(variant_detect_cfg.get("content"))
    variant_lf_repair_cfg = _as_dict(variant_content_cfg.get("lf_exact_repair"))

    lf_score = channel_scores.get("lf")
    if lf_score is None:
        lf_score = image_channel_scores.get("lf", content_payload.get("lf_score"))
    hf_score = channel_scores.get("hf")
    if hf_score is None:
        hf_score = image_channel_scores.get("hf", content_payload.get("hf_score"))
    geo_score = channel_scores.get("geo")
    if geo_score is None:
        geo_score = image_channel_scores.get("geo", geometry_payload.get("geo_score"))

    active_geo_score_source = _pick_first_present_value(
        detect_record,
        [
            "active_geo_score_source",
            "geometry_evidence_payload.active_geo_score_source",
            "attestation.active_geo_score_source",
            "attestation.image_evidence_result.active_geo_score_source",
            "geo_rescue_diagnostics_artifact.active_geo_score_source",
            "_geo_rescue_diagnostics_artifact.active_geo_score_source",
        ],
    )
    geo_repair_enabled = _pick_first_present_value(
        detect_record,
        [
            "geo_repair_enabled",
            "geometry_evidence_payload.geo_repair_enabled",
            "geo_rescue_diagnostics_artifact.geo_repair_enabled",
            "_geo_rescue_diagnostics_artifact.geo_repair_enabled",
        ],
    )
    if geo_repair_enabled is None:
        geo_repair_enabled = variant_geo_repair_cfg.get("enabled")
    geo_repair_active = _pick_first_present_value(
        detect_record,
        [
            "geo_repair_active",
            "geometry_evidence_payload.geo_repair_active",
            "geo_rescue_diagnostics_artifact.geo_repair_active",
            "_geo_rescue_diagnostics_artifact.geo_repair_active",
        ],
    )
    geo_repair_mode = _pick_first_present_value(
        detect_record,
        [
            "geo_repair_mode",
            "geometry_evidence_payload.geo_repair_mode",
            "geo_rescue_diagnostics_artifact.geo_repair_mode",
            "_geo_rescue_diagnostics_artifact.geo_repair_mode",
        ],
    )
    if geo_repair_mode is None:
        geo_repair_mode = variant_geo_repair_cfg.get("mode")
    geo_score_repair_enabled = _pick_first_present_value(
        detect_record,
        [
            "geo_score_repair_enabled",
            "geometry_evidence_payload.geo_score_repair_enabled",
            "geo_rescue_diagnostics_artifact.geo_score_repair_enabled",
            "_geo_rescue_diagnostics_artifact.geo_score_repair_enabled",
        ],
    )
    if geo_score_repair_enabled is None:
        geo_score_repair_enabled = variant_geo_repair_cfg.get("enabled")
    geo_score_repair_active = _pick_first_present_value(
        detect_record,
        [
            "geo_score_repair_active",
            "geometry_evidence_payload.geo_score_repair_active",
            "geo_rescue_diagnostics_artifact.geo_score_repair_active",
            "_geo_rescue_diagnostics_artifact.geo_score_repair_active",
        ],
    )
    geo_score_repair_mode = _pick_first_present_value(
        detect_record,
        [
            "geo_score_repair_mode",
            "geometry_evidence_payload.geo_score_repair_mode",
            "geo_rescue_diagnostics_artifact.geo_score_repair_mode",
            "_geo_rescue_diagnostics_artifact.geo_score_repair_mode",
        ],
    )
    if geo_score_repair_mode is None:
        geo_score_repair_mode = variant_geo_repair_cfg.get("mode")
    geo_repair_direction_classification = _pick_first_present_value(
        detect_record,
        [
            "geo_repair_direction_classification",
            "geometry_evidence_payload.geo_repair_direction_classification",
            "geo_rescue_diagnostics_artifact.geo_repair_direction_classification",
            "_geo_rescue_diagnostics_artifact.geo_repair_direction_classification",
        ],
    )
    protocol_root_cause_classification = _pick_first_present_value(
        detect_record,
        [
            "protocol_root_cause_classification",
            "control_protocol_summary.protocol_root_cause_classification",
            "content_evidence_payload.protocol_root_cause_classification",
            "attestation.protocol_root_cause_classification",
        ],
    )
    formal_exact_evidence_source = _pick_first_present_value(
        detect_record,
        [
            "formal_exact_evidence_source",
            "content_evidence_payload.formal_exact_evidence_source",
            "attestation.formal_exact_evidence_source",
        ],
    )
    lf_exact_repair_enabled = _pick_first_present_value(
        detect_record,
        [
            "lf_exact_repair_enabled",
            "content_evidence_payload.lf_exact_repair_enabled",
        ],
    )
    if lf_exact_repair_enabled is None:
        lf_exact_repair_enabled = variant_lf_repair_cfg.get("enabled")
    lf_exact_repair_applied = _pick_first_present_value(
        detect_record,
        [
            "lf_exact_repair_applied",
            "content_evidence_payload.lf_exact_repair_applied",
        ],
    )
    lf_exact_repair_mode = _pick_first_present_value(
        detect_record,
        [
            "lf_exact_repair_mode",
            "content_evidence_payload.lf_exact_repair_mode",
        ],
    )
    if lf_exact_repair_mode is None:
        lf_exact_repair_mode = variant_lf_repair_cfg.get("mode")

    active_score_source = final_event_decision.get("event_attestation_score_name")
    if not isinstance(active_score_source, str) or not active_score_source:
        active_score_source = image_evidence_result.get("content_attestation_score_name")
    if not isinstance(active_score_source, str) or not active_score_source:
        active_score_source = score_parts.get("content_score_rule_id")
    if not isinstance(active_score_source, str) or not active_score_source:
        active_score_source = "<absent>"

    diagnostics_core = {
        "content_status": content_payload.get("status", content_payload.get("detect_lf_status", "<absent>")),
        "geometry_status": geometry_payload.get("status", "<absent>"),
        "attestation_status": attestation_payload.get("status", "<absent>"),
        "authenticity_status": _as_dict(attestation_payload.get("authenticity_result")).get("status", "<absent>"),
        "image_evidence_status": image_evidence_result.get("status", "<absent>"),
        "event_decision_status": final_event_decision.get("status", "<absent>"),
    }

    return {
        "variant_name": variant.name,
        "variant_suffix": variant.suffix,
        "variant_description": variant.description,
        "variant_group": variant.group,
        "variant_category": variant.category,
        "variant_run_root": str(variant_run_root),
        "base_embed_record_path": str(base_embed_record_path),
        "detect_record_path": str(detect_record_path),
        "input_record_path": str(input_record_path),
        "config_snapshot_path": str(config_snapshot_path),
        "variant_overrides": variant.overrides,
        "attestation_status": attestation_payload.get("status", final_event_decision.get("status", "<absent>")),
        "attestation_result_status": attestation_payload.get("status", "<absent>"),
        "content_attestation_score": image_evidence_result.get(
            "content_attestation_score",
            attestation_payload.get("content_attestation_score"),
        ),
        "event_attestation_score": final_event_decision.get("event_attestation_score"),
        "lf_score": lf_score,
        "hf_score": hf_score,
        "geo_score": geo_score,
        "channel_scores": {"lf": lf_score, "hf": hf_score, "geo": geo_score},
        "channel_scores_lf": lf_score,
        "channel_scores_hf": hf_score,
        "channel_scores_geo": geo_score,
        "content_score": content_payload.get("content_score", content_payload.get("score")),
        "active_score_source": active_score_source,
        "active_geo_score_source": active_geo_score_source,
        "geo_repair_enabled": geo_repair_enabled,
        "geo_repair_active": geo_repair_active,
        "geo_repair_mode": geo_repair_mode,
        "geo_score_repair_enabled": geo_score_repair_enabled,
        "geo_score_repair_active": geo_score_repair_active,
        "geo_score_repair_mode": geo_score_repair_mode,
        "geo_repair_direction_classification": geo_repair_direction_classification,
        "protocol_root_cause_classification": protocol_root_cause_classification,
        "formal_exact_evidence_source": formal_exact_evidence_source,
        "lf_exact_repair_enabled": lf_exact_repair_enabled,
        "lf_exact_repair_applied": lf_exact_repair_applied,
        "lf_exact_repair_mode": lf_exact_repair_mode,
        "diagnostics_core": diagnostics_core,
        "geo_rescue_diagnostics": geo_rescue_diagnostics,
    }


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_row_fields(row: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    return {field: row.get(field) for field in fieldnames}


def _write_compare_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_filter_row_fields(row, fieldnames))


def _resolve_selected_variants(
    variants: List[AblationVariant],
    selected_variant_names: Optional[List[str]],
) -> List[AblationVariant]:
    enabled_variants = [variant for variant in variants if variant.enabled]
    if not selected_variant_names:
        return enabled_variants
    normalized_names = [_require_non_empty_str(name, "selected_variant_name") for name in selected_variant_names]
    selected = [variant for variant in variants if variant.name in normalized_names]
    missing = sorted(set(normalized_names) - {variant.name for variant in selected})
    if missing:
        raise ValueError(f"selected variants are not defined: {missing}")
    return selected


def _resolve_effective_selected_variant_names(
    workflow_cfg: Dict[str, Any],
    selected_variant_names: Optional[List[str]],
) -> Optional[List[str]]:
    if selected_variant_names:
        return selected_variant_names
    notebook_runtime_cfg = _resolve_notebook_runtime_cfg(workflow_cfg)
    configured_names = notebook_runtime_cfg.get("selected_variants")
    if configured_names is None:
        return None
    return _require_string_list(configured_names, f"{WORKFLOW_SECTION_KEY}.notebook_runtime.selected_variants")


def _resolve_run_root(
    workflow_cfg: Dict[str, Any],
    run_root: Path | None,
    run_tag: str | None,
) -> Path:
    if isinstance(run_root, Path):
        return run_root.resolve()
    output_root = _resolve_repo_path(
        _require_non_empty_str(
            workflow_cfg.get("output_root", "outputs/Paper_Ablation_Cuda"),
            f"{WORKFLOW_SECTION_KEY}.output_root",
        )
    )
    effective_tag = run_tag
    if effective_tag is None:
        run_tag_value = workflow_cfg.get("run_tag")
        effective_tag = run_tag_value.strip() if isinstance(run_tag_value, str) and run_tag_value.strip() else None
    if effective_tag is None:
        effective_tag = datetime.now(timezone.utc).strftime("ablation_%Y%m%d_%H%M%S")
    return (output_root / effective_tag).resolve()


def _resolve_reuse_mode(
    workflow_cfg: Dict[str, Any],
    *,
    fresh_run: bool,
    resume: bool,
    reuse_base_embed_record: Path | None,
) -> tuple[bool, bool, Path | None, str]:
    notebook_runtime_cfg = _resolve_notebook_runtime_cfg(workflow_cfg)
    effective_reuse_record = reuse_base_embed_record
    runtime_reuse_mode = notebook_runtime_cfg.get("base_embed_reuse_mode")
    runtime_reuse_path = notebook_runtime_cfg.get("reuse_base_embed_record")

    if effective_reuse_record is None and isinstance(runtime_reuse_path, str) and runtime_reuse_path.strip():
        effective_reuse_record = _resolve_repo_path(runtime_reuse_path)

    effective_fresh_run = bool(fresh_run)
    effective_resume = bool(resume)
    mode_label = "fresh_run" if effective_fresh_run else "resume" if effective_resume else "new_run"

    if not effective_fresh_run and not effective_resume and effective_reuse_record is None:
        normalized_mode = runtime_reuse_mode.strip() if isinstance(runtime_reuse_mode, str) and runtime_reuse_mode.strip() else "new_run"
        if normalized_mode == "fresh_run":
            effective_fresh_run = True
            mode_label = "fresh_run"
        elif normalized_mode == "resume":
            effective_resume = True
            mode_label = "resume"
        elif normalized_mode == "reuse_existing_record":
            if effective_reuse_record is None:
                raise ValueError(
                    "base_embed_reuse_mode=reuse_existing_record requires reuse_base_embed_record to be set"
                )
            mode_label = "reuse_existing_record"
        else:
            mode_label = normalized_mode

    if effective_reuse_record is not None:
        mode_label = "reuse_existing_record"

    return effective_fresh_run, effective_resume, effective_reuse_record, mode_label


def _archive_run_outputs(run_root: Path) -> Path:
    archive_path = run_root.parent / f"{run_root.name}_paper_ablation_outputs.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive_handle:
        for file_path in sorted(run_root.rglob("*")):
            if file_path.is_file():
                archive_handle.write(file_path, arcname=str(file_path.relative_to(run_root.parent)))
    return archive_path


def _prepare_stage_config_for_optional_metrics(
    base_variant_snapshot: Dict[str, Any],
    workflow_cfg: Dict[str, Any],
    variant_run_root: Path,
    *,
    stage_name: str,
) -> Dict[str, Any]:
    detect_rerun_cfg = _require_mapping(workflow_cfg.get("detect_rerun"), f"{WORKFLOW_SECTION_KEY}.detect_rerun")
    cfg_snapshot = cast(Dict[str, Any], json.loads(json.dumps(base_variant_snapshot, ensure_ascii=False)))
    if not isinstance(cfg_snapshot, dict):
        raise TypeError("stage config snapshot must remain dict")
    if stage_name == "calibrate":
        if not bool(detect_rerun_cfg.get("enable_calibration", False)):
            return cfg_snapshot
        records_glob = detect_rerun_cfg.get("calibration_detect_records_glob")
        if not isinstance(records_glob, str) or not records_glob:
            raise ValueError(
                "paper ablation calibration requires paper_ablation_workflow.detect_rerun."
                "calibration_detect_records_glob because single-object detect reruns do not create labeled pairs automatically"
            )
        _apply_nested_override(cfg_snapshot, "calibration.detect_records_glob", records_glob.format(variant_run_root=str(variant_run_root)))
        return cfg_snapshot
    if stage_name == "evaluate":
        if not bool(detect_rerun_cfg.get("enable_evaluate", False)):
            return cfg_snapshot
        records_glob = detect_rerun_cfg.get("evaluate_detect_records_glob")
        if not isinstance(records_glob, str) or not records_glob:
            raise ValueError(
                "paper ablation evaluate requires paper_ablation_workflow.detect_rerun."
                "evaluate_detect_records_glob because single-object detect reruns do not create labeled pairs automatically"
            )
        _apply_nested_override(cfg_snapshot, "evaluate.detect_records_glob", records_glob.format(variant_run_root=str(variant_run_root)))
        reuse_thresholds_artifact = detect_rerun_cfg.get("reuse_thresholds_artifact")
        if isinstance(reuse_thresholds_artifact, str) and reuse_thresholds_artifact:
            _apply_nested_override(cfg_snapshot, "evaluate.thresholds_path", reuse_thresholds_artifact)
        return cfg_snapshot
    raise ValueError(f"unsupported stage_name: {stage_name}")


def run_paper_ablation_workflow(
    config_path: Path,
    run_root: Path | None = None,
    run_tag: str | None = None,
    selected_variant_names: Optional[List[str]] = None,
    fresh_run: bool = False,
    resume: bool = False,
    reuse_base_embed_record: Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    功能：执行 paper 单对象单变量消融工作流。

    Run the paper ablation workflow using one base embed and multiple detect reruns.

    Args:
        config_path: Ablation config YAML path.
        run_root: Optional explicit output root.
        run_tag: Optional run tag used when run_root is absent.
        selected_variant_names: Optional variant-name filter.
        fresh_run: Whether the run root must be empty.
        resume: Whether to continue an existing run root and reuse completed stages.
        reuse_base_embed_record: Optional explicit embed record path reused as the
            shared detect input.
        dry_run: Whether to print commands without executing subprocesses.

    Returns:
        Summary mapping containing key output anchors.

    Raises:
        ValueError: If workflow configuration is invalid.
    """
    resolved_config_path = _resolve_repo_path(config_path)
    cfg_obj = _read_yaml_config(resolved_config_path)
    workflow_cfg = _resolve_workflow_cfg(cfg_obj)
    effective_selected_variant_names = _resolve_effective_selected_variant_names(workflow_cfg, selected_variant_names)
    variants = _resolve_selected_variants(_load_variant_definitions(workflow_cfg), effective_selected_variant_names)
    effective_fresh_run, effective_resume, effective_reuse_base_embed_record, reuse_mode = _resolve_reuse_mode(
        workflow_cfg,
        fresh_run=fresh_run,
        resume=resume,
        reuse_base_embed_record=reuse_base_embed_record,
    )
    effective_run_root = _resolve_run_root(workflow_cfg, run_root, run_tag)
    _ensure_clean_run_root(
        effective_run_root,
        fresh_run=effective_fresh_run,
        resume=effective_resume,
        reuse_base_embed_record=effective_reuse_base_embed_record,
    )
    layout = _resolve_layout(workflow_cfg, effective_run_root)
    summary_fields = _resolve_compare_fields(workflow_cfg, "summary_fields", DEFAULT_COMPARE_SUMMARY_FIELDS)
    table_fields = _resolve_compare_fields(workflow_cfg, "table_fields", DEFAULT_COMPARE_TABLE_FIELDS)
    notebook_runtime_cfg = _resolve_notebook_runtime_cfg(workflow_cfg)

    base_embed_cfg = _require_mapping(workflow_cfg.get("base_embed"), f"{WORKFLOW_SECTION_KEY}.base_embed")
    base_embed_overrides = _normalize_dotted_overrides(base_embed_cfg.get("overrides"), f"{WORKFLOW_SECTION_KEY}.base_embed.overrides")
    base_embed_snapshot = _build_runtime_cfg_snapshot(cfg_obj, base_embed_overrides, resolved_config_path)
    base_embed_cfg_path = layout.config_snapshot_root / "base_embed_config.yaml"
    _write_yaml_file(base_embed_cfg_path, base_embed_snapshot)

    detect_rerun_cfg = _require_mapping(workflow_cfg.get("detect_rerun"), f"{WORKFLOW_SECTION_KEY}.detect_rerun")
    input_record_rel_path = _require_non_empty_str(
        detect_rerun_cfg.get("input_record_rel_path", "records/embed_record.json"),
        f"{WORKFLOW_SECTION_KEY}.detect_rerun.input_record_rel_path",
    )
    base_embed_record_rel_path = _require_non_empty_str(
        base_embed_cfg.get("embed_record_rel_path", input_record_rel_path),
        f"{WORKFLOW_SECTION_KEY}.base_embed.embed_record_rel_path",
    )
    variant_dir_pattern = _require_non_empty_str(
        detect_rerun_cfg.get("variant_dir_pattern", "{suffix}"),
        f"{WORKFLOW_SECTION_KEY}.detect_rerun.variant_dir_pattern",
    )
    allow_resume = bool(base_embed_cfg.get("allow_resume", True))
    allow_reuse_existing_record = bool(base_embed_cfg.get("allow_reuse_existing_record", True))
    allow_detect_only = bool(detect_rerun_cfg.get("allow_detect_only", True))
    reuse_existing_detect_results = bool(detect_rerun_cfg.get("reuse_existing_detect_results", False))

    local_base_embed_record_path = layout.base_embed_root / base_embed_record_rel_path
    if effective_reuse_base_embed_record is not None:
        if not allow_reuse_existing_record:
            raise ValueError("base_embed.allow_reuse_existing_record is false, cannot reuse external embed record")
        base_embed_record_path = _resolve_repo_path(effective_reuse_base_embed_record)
        if not base_embed_record_path.exists() or not base_embed_record_path.is_file():
            raise FileNotFoundError(f"reuse_base_embed_record is missing: {base_embed_record_path}")
        if not allow_detect_only:
            raise ValueError("detect_rerun.allow_detect_only must be true when reuse_base_embed_record is provided")
        base_embed_stage_executed = False
        base_embed_source_mode = "external_embed_record"
    elif effective_resume and allow_resume and local_base_embed_record_path.exists() and local_base_embed_record_path.is_file():
        base_embed_record_path = local_base_embed_record_path
        base_embed_stage_executed = False
        base_embed_source_mode = "resume_existing_base_embed"
    else:
        _run_subprocess(_build_embed_command(layout.base_embed_root, base_embed_cfg_path), _repo_root, dry_run)
        base_embed_record_path = local_base_embed_record_path
        if not dry_run and (not base_embed_record_path.exists() or not base_embed_record_path.is_file()):
            raise FileNotFoundError(f"base embed record missing after embed stage: {base_embed_record_path}")
        base_embed_stage_executed = True
        base_embed_source_mode = "new_embed"

    base_input_record_path = base_embed_record_path

    variant_rows: List[Dict[str, Any]] = []
    variant_manifest_entries: List[Dict[str, Any]] = []
    for variant in variants:
        variant_rel_dir = variant_dir_pattern.format(name=variant.name, suffix=variant.suffix)
        variant_run_root = layout.variants_root / variant_rel_dir
        variant_run_root.mkdir(parents=True, exist_ok=True)

        variant_snapshot = _build_runtime_cfg_snapshot(cfg_obj, variant.overrides, resolved_config_path)
        variant_cfg_path = layout.config_snapshot_root / "variants" / f"{variant.suffix}.yaml"
        _write_yaml_file(variant_cfg_path, variant_snapshot)

        detect_record_path = variant_run_root / "records" / "detect_record.json"
        reuse_existing_detect_record = bool(
            effective_resume and reuse_existing_detect_results and detect_record_path.exists() and detect_record_path.is_file()
        )
        if not reuse_existing_detect_record:
            _run_subprocess(
                _build_detect_command(variant_run_root, variant_cfg_path, base_input_record_path),
                _repo_root,
                dry_run,
            )

        calibrate_cfg_path = None
        evaluate_cfg_path = None
        if bool(detect_rerun_cfg.get("enable_calibration", False)):
            calibrate_snapshot = _prepare_stage_config_for_optional_metrics(
                variant_snapshot,
                workflow_cfg,
                variant_run_root,
                stage_name="calibrate",
            )
            calibrate_cfg_path = layout.config_snapshot_root / "variants" / f"{variant.suffix}_calibrate.yaml"
            _write_yaml_file(calibrate_cfg_path, calibrate_snapshot)
            _run_subprocess(_build_calibrate_command(variant_run_root, calibrate_cfg_path), _repo_root, dry_run)
        if bool(detect_rerun_cfg.get("enable_evaluate", False)):
            evaluate_snapshot = _prepare_stage_config_for_optional_metrics(
                variant_snapshot,
                workflow_cfg,
                variant_run_root,
                stage_name="evaluate",
            )
            evaluate_cfg_path = layout.config_snapshot_root / "variants" / f"{variant.suffix}_evaluate.yaml"
            _write_yaml_file(evaluate_cfg_path, evaluate_snapshot)
            _run_subprocess(_build_evaluate_command(variant_run_root, evaluate_cfg_path), _repo_root, dry_run)

        if dry_run:
            detect_record_payload = {}
        else:
            detect_record_payload = _load_json_dict(detect_record_path)
        compare_row = _extract_variant_compare_row(
            variant=variant,
            variant_run_root=variant_run_root,
            base_embed_record_path=base_embed_record_path,
            detect_record_path=detect_record_path,
            input_record_path=base_input_record_path,
            config_snapshot_path=variant_cfg_path,
            variant_cfg_snapshot=variant_snapshot,
            detect_record=detect_record_payload,
        )
        variant_rows.append(compare_row)
        variant_manifest_entries.append(
            {
                "name": variant.name,
                "suffix": variant.suffix,
                "description": variant.description,
                "variant_run_root": str(variant_run_root),
                "input_record_path": str(base_input_record_path),
                "detect_record_path": str(detect_record_path),
                "config_snapshot_path": str(variant_cfg_path),
                "calibrate_config_snapshot_path": str(calibrate_cfg_path) if calibrate_cfg_path is not None else None,
                "evaluate_config_snapshot_path": str(evaluate_cfg_path) if evaluate_cfg_path is not None else None,
                "detect_stage_executed": not reuse_existing_detect_record,
                "reused_existing_detect_record": reuse_existing_detect_record,
                "group": variant.group,
                "category": variant.category,
                "overrides": variant.overrides,
            }
        )

    archive_path = None
    if bool(notebook_runtime_cfg.get("package_zip", False)):
        archive_path = _archive_run_outputs(layout.run_root)

    manifest_obj: Dict[str, Any] = {
        "schema_version": NOTEBOOK_CONFIG_TEMPLATE_VERSION,
        "config_path": str(resolved_config_path),
        "run_root": str(layout.run_root),
        "execution_mode": {
            "requested": {
                "fresh_run": bool(fresh_run),
                "resume": bool(resume),
                "reuse_base_embed_record": (
                    str(reuse_base_embed_record) if reuse_base_embed_record is not None else None
                ),
            },
            "effective": {
                "fresh_run": effective_fresh_run,
                "resume": effective_resume,
                "reuse_mode": reuse_mode,
                "reuse_base_embed_record": (
                    str(effective_reuse_base_embed_record)
                    if effective_reuse_base_embed_record is not None
                    else None
                ),
            },
        },
        "base_embed": {
            "run_root": str(layout.base_embed_root),
            "embed_record_path": str(base_embed_record_path),
            "config_snapshot_path": str(base_embed_cfg_path),
            "stage_executed": base_embed_stage_executed,
            "source_mode": base_embed_source_mode,
            "reuse_mode": reuse_mode,
            "reuse_source_path": str(effective_reuse_base_embed_record) if effective_reuse_base_embed_record is not None else None,
            "overrides": base_embed_overrides,
        },
        "detect_rerun": {
            "input_record_path": str(base_input_record_path),
            "allow_detect_only": allow_detect_only,
            "reuse_existing_detect_results": reuse_existing_detect_results,
            "enable_calibration": bool(detect_rerun_cfg.get("enable_calibration", False)),
            "enable_evaluate": bool(detect_rerun_cfg.get("enable_evaluate", False)),
            "reuse_thresholds_artifact": detect_rerun_cfg.get("reuse_thresholds_artifact"),
        },
        "compare": {
            "summary_fields": summary_fields,
            "table_fields": table_fields,
        },
        "notebook_runtime": notebook_runtime_cfg,
        "archive_path": str(archive_path) if archive_path is not None else None,
        "variants": variant_manifest_entries,
    }
    compare_summary_obj: Dict[str, Any] = {
        "schema_version": "paper_ablation_compare_summary_v2",
        "run_root": str(layout.run_root),
        "base_embed_run_root": str(layout.base_embed_root),
        "base_embed_record_path": str(base_embed_record_path),
        "reuse_mode": reuse_mode,
        "archive_path": str(archive_path) if archive_path is not None else None,
        "variant_count": len(variant_rows),
        "summary_fields": summary_fields,
        "table_fields": table_fields,
        "variants": [_filter_row_fields(row, summary_fields) for row in variant_rows],
    }

    manifest_path = layout.compare_root / "ablation_manifest.json"
    compare_summary_path = layout.compare_root / "ablation_compare_summary.json"
    compare_table_path = layout.compare_root / "ablation_compare_table.csv"
    _write_json(manifest_path, manifest_obj)
    _write_json(compare_summary_path, compare_summary_obj)
    _write_compare_csv(compare_table_path, variant_rows, table_fields)

    result: Dict[str, Any] = {
        "run_root": str(layout.run_root),
        "base_embed_run_root": str(layout.base_embed_root),
        "base_embed_record_path": str(base_embed_record_path),
        "reuse_mode": reuse_mode,
        "manifest_path": str(manifest_path),
        "compare_summary_path": str(compare_summary_path),
        "compare_table_path": str(compare_table_path),
        "archive_path": str(archive_path) if archive_path is not None else None,
        "variant_names": [variant.name for variant in variants],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-base-embed paper ablation workflow with multiple detect reruns."
    )
    parser.add_argument(
        "--config",
        default="configs/ablation/paper_ablation_cuda.yaml",
        help="Ablation config YAML path.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional explicit output root.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional run tag appended under workflow output_root when --run-root is absent.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant name to execute. Can be repeated.",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Require run_root to be empty before execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed base embed and detect stages under an existing run_root when possible.",
    )
    parser.add_argument(
        "--reuse-base-embed-record",
        default=None,
        help="Explicit existing embed_record.json path reused as the shared detect input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing subprocesses.",
    )
    return parser


def main() -> int:
    """
    功能：CLI 主入口。

    Execute the paper ablation workflow CLI entry point.

    Args:
        None.

    Returns:
        Process exit code.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        run_paper_ablation_workflow(
            config_path=_resolve_repo_path(args.config),
            run_root=_resolve_repo_path(args.run_root) if isinstance(args.run_root, str) and args.run_root.strip() else None,
            run_tag=args.run_tag,
            selected_variant_names=args.variant,
            fresh_run=bool(args.fresh_run),
            resume=bool(args.resume),
            reuse_base_embed_record=(
                _resolve_repo_path(args.reuse_base_embed_record)
                if isinstance(args.reuse_base_embed_record, str) and args.reuse_base_embed_record.strip()
                else None
            ),
            dry_run=bool(args.dry_run),
        )
        return 0
    except Exception as exc:
        print(f"[paper_ablation] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
