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
NOTEBOOK_CONFIG_TEMPLATE_VERSION = "paper_ablation_workflow_v1"


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
    _load_variant_definitions(workflow_cfg)
    return workflow_cfg


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


def _build_runtime_cfg_snapshot(base_cfg_obj: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = cast(Dict[str, Any], json.loads(json.dumps(base_cfg_obj, ensure_ascii=False)))
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must remain dict")
    snapshot.pop(WORKFLOW_SECTION_KEY, None)
    for dotted_path, override_value in overrides.items():
        _apply_nested_override(snapshot, dotted_path, override_value)
    return snapshot


def _write_yaml_file(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _ensure_clean_run_root(run_root: Path) -> None:
    if run_root.exists():
        if not run_root.is_dir():
            raise ValueError(f"run_root exists and is not a directory: {run_root}")
        if any(run_root.iterdir()):
            raise ValueError(f"run_root must be empty for paper ablation workflow: {run_root}")
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


def _build_embed_command(run_root: Path, config_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "main.cli.run_embed",
        "--out",
        str(run_root),
        "--config",
        str(config_path),
    ]


def _build_detect_command(run_root: Path, config_path: Path, input_record_path: Path) -> List[str]:
    return [
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
    detect_record_path: Path,
    input_record_path: Path,
    config_snapshot_path: Path,
    detect_record: Dict[str, Any],
) -> Dict[str, Any]:
    attestation_payload = _as_dict(detect_record.get("attestation"))
    image_evidence_result = _as_dict(attestation_payload.get("image_evidence_result"))
    final_event_decision = _as_dict(attestation_payload.get("final_event_attested_decision"))
    content_payload = _as_dict(detect_record.get("content_evidence_payload"))
    geometry_payload = _as_dict(detect_record.get("geometry_evidence_payload"))
    score_parts = _as_dict(content_payload.get("score_parts"))
    channel_scores = _as_dict(attestation_payload.get("channel_scores"))
    image_channel_scores = _as_dict(image_evidence_result.get("channel_scores"))

    lf_score = channel_scores.get("lf")
    if lf_score is None:
        lf_score = image_channel_scores.get("lf", content_payload.get("lf_score"))
    hf_score = channel_scores.get("hf")
    if hf_score is None:
        hf_score = image_channel_scores.get("hf", content_payload.get("hf_score"))
    geo_score = channel_scores.get("geo")
    if geo_score is None:
        geo_score = image_channel_scores.get("geo", geometry_payload.get("geo_score"))

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
        "variant_run_root": str(variant_run_root),
        "detect_record_path": str(detect_record_path),
        "input_record_path": str(input_record_path),
        "config_snapshot_path": str(config_snapshot_path),
        "variant_overrides": variant.overrides,
        "attestation_status": attestation_payload.get("status", final_event_decision.get("status", "<absent>")),
        "content_attestation_score": image_evidence_result.get(
            "content_attestation_score",
            attestation_payload.get("content_attestation_score"),
        ),
        "event_attestation_score": final_event_decision.get("event_attestation_score"),
        "lf_score": lf_score,
        "hf_score": hf_score,
        "geo_score": geo_score,
        "content_score": content_payload.get("content_score", content_payload.get("score")),
        "active_score_source": active_score_source,
        "diagnostics_core": diagnostics_core,
    }


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_compare_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_name",
        "variant_suffix",
        "attestation_status",
        "content_attestation_score",
        "event_attestation_score",
        "lf_score",
        "hf_score",
        "geo_score",
        "content_score",
        "active_score_source",
        "variant_run_root",
        "detect_record_path",
        "input_record_path",
        "config_snapshot_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


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
        dry_run: Whether to print commands without executing subprocesses.

    Returns:
        Summary mapping containing key output anchors.

    Raises:
        ValueError: If workflow configuration is invalid.
    """
    resolved_config_path = _resolve_repo_path(config_path)
    cfg_obj = _read_yaml_config(resolved_config_path)
    workflow_cfg = _resolve_workflow_cfg(cfg_obj)
    variants = _resolve_selected_variants(_load_variant_definitions(workflow_cfg), selected_variant_names)
    effective_run_root = _resolve_run_root(workflow_cfg, run_root, run_tag)
    _ensure_clean_run_root(effective_run_root)
    layout = _resolve_layout(workflow_cfg, effective_run_root)

    base_embed_cfg = _require_mapping(workflow_cfg.get("base_embed"), f"{WORKFLOW_SECTION_KEY}.base_embed")
    base_embed_overrides = _normalize_dotted_overrides(base_embed_cfg.get("overrides"), f"{WORKFLOW_SECTION_KEY}.base_embed.overrides")
    base_embed_snapshot = _build_runtime_cfg_snapshot(cfg_obj, base_embed_overrides)
    base_embed_cfg_path = layout.config_snapshot_root / "base_embed_config.yaml"
    _write_yaml_file(base_embed_cfg_path, base_embed_snapshot)

    _run_subprocess(_build_embed_command(layout.base_embed_root, base_embed_cfg_path), _repo_root, dry_run)
    base_embed_record_path = layout.base_embed_root / "records" / "embed_record.json"
    if not dry_run and (not base_embed_record_path.exists() or not base_embed_record_path.is_file()):
        raise FileNotFoundError(f"base embed record missing after embed stage: {base_embed_record_path}")

    detect_rerun_cfg = _require_mapping(workflow_cfg.get("detect_rerun"), f"{WORKFLOW_SECTION_KEY}.detect_rerun")
    input_record_rel_path = _require_non_empty_str(
        detect_rerun_cfg.get("input_record_rel_path", "records/embed_record.json"),
        f"{WORKFLOW_SECTION_KEY}.detect_rerun.input_record_rel_path",
    )
    variant_dir_pattern = _require_non_empty_str(
        detect_rerun_cfg.get("variant_dir_pattern", "{suffix}"),
        f"{WORKFLOW_SECTION_KEY}.detect_rerun.variant_dir_pattern",
    )
    base_input_record_path = layout.base_embed_root / input_record_rel_path

    variant_rows: List[Dict[str, Any]] = []
    variant_manifest_entries: List[Dict[str, Any]] = []
    for variant in variants:
        variant_rel_dir = variant_dir_pattern.format(name=variant.name, suffix=variant.suffix)
        variant_run_root = layout.variants_root / variant_rel_dir
        variant_run_root.mkdir(parents=True, exist_ok=True)

        variant_snapshot = _build_runtime_cfg_snapshot(cfg_obj, variant.overrides)
        variant_cfg_path = layout.config_snapshot_root / "variants" / f"{variant.suffix}.yaml"
        _write_yaml_file(variant_cfg_path, variant_snapshot)

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

        detect_record_path = variant_run_root / "records" / "detect_record.json"
        if dry_run:
            detect_record_payload = {}
        else:
            detect_record_payload = _load_json_dict(detect_record_path)
        compare_row = _extract_variant_compare_row(
            variant=variant,
            variant_run_root=variant_run_root,
            detect_record_path=detect_record_path,
            input_record_path=base_input_record_path,
            config_snapshot_path=variant_cfg_path,
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
                "overrides": variant.overrides,
            }
        )

    manifest_obj: Dict[str, Any] = {
        "schema_version": NOTEBOOK_CONFIG_TEMPLATE_VERSION,
        "config_path": str(resolved_config_path),
        "run_root": str(layout.run_root),
        "base_embed": {
            "run_root": str(layout.base_embed_root),
            "embed_record_path": str(base_embed_record_path),
            "config_snapshot_path": str(base_embed_cfg_path),
            "overrides": base_embed_overrides,
        },
        "detect_rerun": {
            "input_record_path": str(base_input_record_path),
            "enable_calibration": bool(detect_rerun_cfg.get("enable_calibration", False)),
            "enable_evaluate": bool(detect_rerun_cfg.get("enable_evaluate", False)),
            "reuse_thresholds_artifact": detect_rerun_cfg.get("reuse_thresholds_artifact"),
        },
        "variants": variant_manifest_entries,
    }
    compare_summary_obj: Dict[str, Any] = {
        "schema_version": "paper_ablation_compare_summary_v1",
        "run_root": str(layout.run_root),
        "base_embed_run_root": str(layout.base_embed_root),
        "base_embed_record_path": str(base_embed_record_path),
        "variant_count": len(variant_rows),
        "variants": variant_rows,
    }

    manifest_path = layout.compare_root / "ablation_manifest.json"
    compare_summary_path = layout.compare_root / "ablation_compare_summary.json"
    compare_table_path = layout.compare_root / "ablation_compare_table.csv"
    _write_json(manifest_path, manifest_obj)
    _write_json(compare_summary_path, compare_summary_obj)
    _write_compare_csv(compare_table_path, variant_rows)

    result: Dict[str, Any] = {
        "run_root": str(layout.run_root),
        "base_embed_run_root": str(layout.base_embed_root),
        "base_embed_record_path": str(base_embed_record_path),
        "manifest_path": str(manifest_path),
        "compare_summary_path": str(compare_summary_path),
        "compare_table_path": str(compare_table_path),
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
            dry_run=bool(args.dry_run),
        )
        return 0
    except Exception as exc:
        print(f"[paper_ablation] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
