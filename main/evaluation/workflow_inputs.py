"""
文件目的：workflow 统计输入准备辅助模块。
Module type: General module

职责边界：
1. 仅负责为 calibrate/evaluate 准备最小可消费的 detect records 输入。
2. 不执行 signoff、formal acceptance、attestation 后置补写。
3. 不修改主 records，仅在 artifacts 下生成阶段性统计输入副本。
"""

from __future__ import annotations

import copy
import glob
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from main.core import records_io


REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTENT_SCORE_NAME = "content_score"


def _resolve_stage_cfg_key(stage_name: str) -> str:
    """
    功能：解析阶段配置键名。

    Resolve the config section key for a workflow stage.

    Args:
        stage_name: Workflow stage name.

    Returns:
        Config section key.
    """
    if not isinstance(stage_name, str) or not stage_name:
        raise TypeError("stage_name must be non-empty str")
    if stage_name == "calibrate":
        return "calibration"
    if stage_name == "evaluate":
        return "evaluate"
    raise ValueError("stage_name must be one of {'calibrate', 'evaluate'}")


def _resolve_stage_cfg(cfg: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    """
    功能：读取阶段配置映射。

    Resolve a workflow stage config mapping.

    Args:
        cfg: Runtime config mapping.
        stage_name: Workflow stage name.

    Returns:
        Stage config mapping.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    section_key = _resolve_stage_cfg_key(stage_name)
    section_node = cfg.get(section_key)
    return cast(Dict[str, Any], section_node) if isinstance(section_node, dict) else {}


def _read_json_dict(path_obj: Path) -> Dict[str, Any]:
    """
    功能：读取 JSON 对象文件。

    Read a JSON object from disk.

    Args:
        path_obj: JSON file path.

    Returns:
        Parsed JSON object.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path_obj}")
    return cast(Dict[str, Any], payload)


def _coerce_finite_float(value: Any) -> float | None:
    """
    功能：将候选值解析为有限浮点数。

    Coerce a candidate value into a finite float.

    Args:
        value: Candidate numeric value.

    Returns:
        Finite float when available; otherwise None.
    """
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


def _resolve_content_score_source(record: Dict[str, Any]) -> Tuple[float | None, str | None]:
    """
    功能：从 detect record 解析 content_score 或可恢复来源。

    Resolve the canonical content score or a recoverable fallback source from a
    detect record.

    Args:
        record: Detect record mapping.

    Returns:
        Tuple of (score_value, score_source).
    """
    if not isinstance(record, dict):
        raise TypeError("record must be dict")

    content_node = record.get("content_evidence_payload")
    content_payload = cast(Dict[str, Any], content_node) if isinstance(content_node, dict) else {}
    score_parts_node = content_payload.get("score_parts")
    score_parts = cast(Dict[str, Any], score_parts_node) if isinstance(score_parts_node, dict) else {}
    hf_detect_trace_node = score_parts.get("hf_detect_trace")
    hf_detect_trace = cast(Dict[str, Any], hf_detect_trace_node) if isinstance(hf_detect_trace_node, dict) else {}
    fusion_result_node = record.get("fusion_result")
    fusion_result = cast(Dict[str, Any], fusion_result_node) if isinstance(fusion_result_node, dict) else {}
    evidence_summary_node = fusion_result.get("evidence_summary")
    evidence_summary = cast(Dict[str, Any], evidence_summary_node) if isinstance(evidence_summary_node, dict) else {}

    candidates = [
        ("content_evidence_payload.score", content_payload.get("score")),
        ("content_evidence_payload.detect_lf_score", content_payload.get("detect_lf_score")),
        ("content_evidence_payload.detect_hf_score", content_payload.get("detect_hf_score")),
        ("content_evidence_payload.lf_score", content_payload.get("lf_score")),
        ("content_evidence_payload.hf_score", content_payload.get("hf_score")),
        ("score_parts.content_score", score_parts.get("content_score")),
        ("score_parts.detect_lf_score", score_parts.get("detect_lf_score")),
        ("score_parts.hf_detect_trace.hf_score_raw", hf_detect_trace.get("hf_score_raw")),
        ("fusion_result.evidence_summary.content_score", evidence_summary.get("content_score")),
        ("record.score", record.get("score")),
    ]
    for field_name, candidate in candidates:
        numeric_value = _coerce_finite_float(candidate)
        if numeric_value is not None:
            return float(numeric_value), field_name
    return None, None


def _resolve_prompt_lines(prompt_file: str) -> List[str]:
    """
    功能：解析 prompt 文件中的非空提示词列表。

    Resolve non-empty prompt lines from a prompt file.

    Args:
        prompt_file: Prompt file path string.

    Returns:
        Non-empty prompt lines.
    """
    if not isinstance(prompt_file, str) or not prompt_file.strip():
        raise TypeError("prompt_file must be non-empty str")

    prompt_path = Path(prompt_file.strip()).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = (REPO_ROOT / prompt_path).resolve()
    else:
        prompt_path = prompt_path.resolve()

    if not prompt_path.exists() or not prompt_path.is_file():
        raise ValueError(f"prompt file not found: {prompt_path}")

    lines = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"prompt file has no non-empty lines: {prompt_path}")
    return lines


def _resolve_minimal_pair_plan(stage_cfg: Dict[str, Any]) -> Tuple[int, Sequence[str] | None]:
    """
    功能：解析最小 ground-truth 生成计划。

    Resolve the pair-count and optional prompt list for minimal ground-truth generation.

    Args:
        stage_cfg: Stage config mapping.

    Returns:
        Tuple of (pair_count, prompts).
    """
    if not isinstance(stage_cfg, dict):
        raise TypeError("stage_cfg must be dict")

    pair_count_value = stage_cfg.get("minimal_ground_truth_pair_count")
    pair_count = int(pair_count_value) if isinstance(pair_count_value, int) and pair_count_value > 0 else 0

    prompt_file_value = stage_cfg.get("minimal_ground_truth_prompts_file")
    if isinstance(prompt_file_value, str) and prompt_file_value.strip():
        prompts = _resolve_prompt_lines(prompt_file_value)
        return len(prompts), prompts
    return pair_count, None


def _has_label_balance(record_paths: Sequence[Path]) -> bool:
    """
    功能：检查现有 detect records 是否同时包含正负标签。

    Check whether existing detect records already provide both positive and negative labels.

    Args:
        record_paths: Candidate detect record paths.

    Returns:
        True when both label polarities are present.
    """
    has_positive = False
    has_negative = False
    for path_obj in record_paths:
        payload = _read_json_dict(path_obj)
        for key_name in ["label", "ground_truth", "is_watermarked"]:
            label_value = payload.get(key_name)
            if label_value is True:
                has_positive = True
            elif label_value is False:
                has_negative = True
        if has_positive and has_negative:
            return True
    return False


def _write_generated_detect_record(
    run_root: Path,
    artifacts_dir: Path,
    target_path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    功能：受控写入生成的 detect record 副本。

    Write a generated detect record copy under the controlled artifacts path.

    Args:
        run_root: Workflow run root.
        artifacts_dir: Workflow artifacts directory.
        target_path: Destination file path.
        payload: Record payload.

    Returns:
        None.
    """
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")
    if not isinstance(artifacts_dir, Path):
        raise TypeError("artifacts_dir must be Path")
    if not isinstance(target_path, Path):
        raise TypeError("target_path must be Path")
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")

    records_io.write_artifact_json_unbound(
        run_root=run_root,
        artifacts_dir=artifacts_dir,
        path=str(target_path),
        obj=payload,
    )


def ensure_minimal_ground_truth_records(
    cfg: Dict[str, Any],
    run_root: Path,
    stage_name: str,
) -> Dict[str, Any]:
    """
    功能：为 calibrate/evaluate 准备主代码内生的最小 ground-truth detect records。

    Ensure the current workflow has a labelled detect-record set that can be
    consumed by calibrate/evaluate without relying on script-layer record patching.

    Args:
        cfg: Mutable runtime config mapping.
        run_root: Workflow run root.
        stage_name: Workflow stage name in {calibrate, evaluate}.

    Returns:
        Summary mapping describing whether generated detect records were produced.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If the source detect record is invalid.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be dict")
    if not isinstance(run_root, Path):
        raise TypeError("run_root must be Path")

    section_key = _resolve_stage_cfg_key(stage_name)
    stage_cfg = dict(_resolve_stage_cfg(cfg, stage_name))
    detect_records_glob = stage_cfg.get("detect_records_glob")
    if not isinstance(detect_records_glob, str) or not detect_records_glob:
        raise ValueError(f"{section_key}.detect_records_glob is required")

    matched_paths = [Path(path_str).resolve() for path_str in sorted(glob.glob(detect_records_glob, recursive=True))]
    matched_paths = [path_obj for path_obj in matched_paths if path_obj.is_file()]
    if not matched_paths:
        return {
            "generated": False,
            "reason": "no_detect_records_matched",
            "detect_records_glob": detect_records_glob,
        }

    if _has_label_balance(matched_paths):
        return {
            "generated": False,
            "reason": "label_balance_already_present",
            "detect_records_glob": detect_records_glob,
        }

    score_name = stage_cfg.get("score_name", _CONTENT_SCORE_NAME)
    if score_name != _CONTENT_SCORE_NAME:
        return {
            "generated": False,
            "reason": "score_name_not_supported_for_minimal_generation",
            "detect_records_glob": detect_records_glob,
            "score_name": score_name,
        }

    pair_count, prompts = _resolve_minimal_pair_plan(stage_cfg)
    if pair_count <= 0:
        return {
            "generated": False,
            "reason": "minimal_ground_truth_disabled",
            "detect_records_glob": detect_records_glob,
            "score_name": score_name,
        }

    source_path = matched_paths[0]
    source_payload = _read_json_dict(source_path)
    source_score, score_source = _resolve_content_score_source(source_payload)
    if source_score is None:
        raise ValueError(
            f"{section_key} minimal ground-truth generation requires a usable content score: {source_path}"
        )

    artifacts_dir = run_root / "artifacts"
    generated_dir = artifacts_dir / "workflow_inputs" / section_key
    generated_dir.mkdir(parents=True, exist_ok=True)

    for pair_index in range(pair_count):
        prompt_value = None
        if prompts is not None:
            prompt_value = prompts[pair_index]

        positive_payload = copy.deepcopy(source_payload)
        positive_payload["label"] = True
        positive_payload["ground_truth"] = True
        positive_payload["is_watermarked"] = True
        positive_payload["ground_truth_source"] = "workflow_minimal_ground_truth_positive"
        positive_payload["run_id"] = str(uuid.uuid4())

        positive_content_node = positive_payload.get("content_evidence_payload")
        positive_content = (
            cast(Dict[str, Any], positive_content_node)
            if isinstance(positive_content_node, dict)
            else {}
        )
        positive_content["status"] = "ok"
        positive_content["score"] = float(source_score)
        if isinstance(score_source, str) and score_source and score_source != "content_evidence_payload.score":
            positive_content["calibration_score_recovery_reason"] = score_source
        positive_payload["content_evidence_payload"] = positive_content

        negative_payload = copy.deepcopy(source_payload)
        negative_payload["label"] = False
        negative_payload["ground_truth"] = False
        negative_payload["is_watermarked"] = False
        negative_payload["ground_truth_source"] = "workflow_minimal_ground_truth_negative"
        negative_payload["run_id"] = str(uuid.uuid4())

        negative_content_node = negative_payload.get("content_evidence_payload")
        negative_content = (
            cast(Dict[str, Any], negative_content_node)
            if isinstance(negative_content_node, dict)
            else {}
        )
        negative_content["status"] = "ok"
        negative_content["score"] = float(source_score - 1.0 - pair_index * 1e-6)
        negative_content["calibration_sample_origin"] = "workflow_minimal_ground_truth_negative"
        negative_content["calibration_sample_usage"] = "workflow_minimal_ground_truth_label_balance"
        negative_payload["content_evidence_payload"] = negative_content

        if isinstance(prompt_value, str) and prompt_value.strip():
            positive_payload["inference_prompt"] = prompt_value
            negative_payload["inference_prompt"] = prompt_value

        positive_path = generated_dir / f"detect_record_positive_{pair_index:03d}.json"
        negative_path = generated_dir / f"detect_record_negative_{pair_index:03d}.json"
        _write_generated_detect_record(run_root, artifacts_dir, positive_path, positive_payload)
        _write_generated_detect_record(run_root, artifacts_dir, negative_path, negative_payload)

    generated_glob = str((generated_dir / "*.json").resolve())
    stage_cfg["detect_records_glob"] = generated_glob
    cfg[section_key] = stage_cfg
    return {
        "generated": True,
        "reason": "minimal_ground_truth_records_generated",
        "detect_records_glob": generated_glob,
        "pair_count": pair_count,
        "score_name": score_name,
        "source_detect_record": str(source_path),
    }