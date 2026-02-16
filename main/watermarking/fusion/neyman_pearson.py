"""
Neyman-Pearson 阈值基线实现与审计口径

功能说明：
- 提供 Neyman-Pearson 阈值基线实现，包含阈值规范构建、元信息构建、摘要计算和阈值选择逻辑。
- 定义稳定的阈值元信息 schema 和校验逻辑，确保元信息的一致性和完整性。
- 按冻结口径加载阈值的函数，包含严格的输入校验和错误处理，确保加载过程的健壮性。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from main.core import digests


RULE_ID = "fusion_neyman_pearson_v1"
RULE_VERSION = "v1"

REQUIRED_METADATA_KEYS = [
    "method",
    "null_source",
    "n_null",
    "calibration_date",
    "quantile_method",
    "target_fprs"
]


def build_thresholds_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造阈值基线 spec。

    Build a deterministic thresholds spec from config.

    Args:
        cfg: Configuration mapping.

    Returns:
        Thresholds spec mapping with audit anchors.

    Raises:
        TypeError: If cfg is invalid.
        ValueError: If target_fpr is invalid.
    """
    if not isinstance(cfg, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")

    target_fpr = _extract_target_fpr(cfg)
    thresholds_spec = {
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "target_fpr": float(target_fpr),
        "fpr_key": format_fpr_key_canonical(target_fpr),
        "method": "neyman_pearson_v1"
    }
    digests.normalize_for_digest(thresholds_spec)
    return thresholds_spec


def build_threshold_metadata(thresholds_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    功能：构造阈值元信息基线结构。

    Build deterministic threshold metadata with a fixed schema.

    Args:
        thresholds_spec: Thresholds spec mapping.

    Returns:
        Threshold metadata mapping.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If target_fpr is invalid.
    """
    if not isinstance(thresholds_spec, dict):
        # thresholds_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("thresholds_spec must be dict")

    target_fpr = thresholds_spec.get("target_fpr")
    if not isinstance(target_fpr, (int, float)):
        raise ValueError("thresholds_spec.target_fpr must be number")

    metadata = {
        "method": "neyman_pearson_v1",
        "null_source": "calibration_null",
        "n_null": 0,
        "calibration_date": "1970-01-01",
        "quantile_method": "nearest",
        "target_fprs": [float(target_fpr)]
    }
    validate_thresholds_metadata(metadata)
    digests.normalize_for_digest(metadata)
    return metadata


def compute_thresholds_digest(thresholds_spec: Dict[str, Any]) -> str:
    """
    功能：计算 thresholds_digest。

    Compute thresholds digest using canonical semantic digest.

    Args:
        thresholds_spec: Thresholds spec mapping.

    Returns:
        Thresholds digest string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If digest output is invalid.
    """
    if not isinstance(thresholds_spec, dict):
        # thresholds_spec 类型不符合预期，必须 fail-fast。
        raise TypeError("thresholds_spec must be dict")
    digests.normalize_for_digest(thresholds_spec)
    digest_value = digests.semantic_digest(thresholds_spec)
    if not isinstance(digest_value, str) or not digest_value:
        raise ValueError("thresholds_digest must be non-empty str")
    return digest_value


def compute_threshold_metadata_digest(metadata: Dict[str, Any]) -> str:
    """
    功能：计算 threshold_metadata_digest。

    Compute threshold metadata digest using canonical semantic digest.

    Args:
        metadata: Threshold metadata mapping.

    Returns:
        Threshold metadata digest string.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If digest output is invalid.
    """
    if not isinstance(metadata, dict):
        # metadata 类型不符合预期，必须 fail-fast。
        raise TypeError("metadata must be dict")
    validate_thresholds_metadata(metadata)
    digest_value = digests.semantic_digest(metadata)
    if not isinstance(digest_value, str) or not digest_value:
        raise ValueError("threshold_metadata_digest must be non-empty str")
    return digest_value


def select_thresholds_np(thresholds: Dict[str, Any], target_fpr: float) -> Dict[str, Any]:
    """
    功能：按 NP 基线逻辑选择阈值。

    Select thresholds with a deterministic baseline policy.

    Args:
        thresholds: Thresholds mapping loaded from file.
        target_fpr: Target false positive rate.

    Returns:
        Threshold info mapping with audit anchors.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If target_fpr is invalid.
    """
    if not isinstance(thresholds, dict):
        # thresholds 类型不符合预期，必须 fail-fast。
        raise TypeError("thresholds must be dict")
    if not isinstance(target_fpr, (int, float)):
        # target_fpr 类型不符合预期，必须 fail-fast。
        raise TypeError("target_fpr must be number")

    thresholds_spec = {
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "target_fpr": float(target_fpr),
        "fpr_key": format_fpr_key_canonical(float(target_fpr)),
        "method": "neyman_pearson_v1"
    }
    thresholds_digest = compute_thresholds_digest(thresholds_spec)
    metadata = build_threshold_metadata(thresholds_spec)
    metadata_digest = compute_threshold_metadata_digest(metadata)

    fpr_key = thresholds_spec["fpr_key"]
    threshold_value = _lookup_threshold_value(thresholds, fpr_key)

    return {
        "rule_id": RULE_ID,
        "rule_version": RULE_VERSION,
        "target_fpr": float(target_fpr),
        "threshold_key_used": fpr_key,
        "threshold_value_used": threshold_value,
        "thresholds_digest": thresholds_digest,
        "threshold_metadata_digest": metadata_digest,
        "metadata": metadata,
        "fpr_key_collision_check": True
    }


def format_fpr_key_canonical(target_fpr: float) -> str:
    """
    功能：格式化 FPR key。

    Format target_fpr as canonical FPR key string.

    Args:
        target_fpr: Target false positive rate.

    Returns:
        Canonical FPR key string.

    Raises:
        TypeError: If target_fpr is invalid.
        ValueError: If target_fpr is non-positive.
    """
    if not isinstance(target_fpr, (int, float)):
        # target_fpr 类型不符合预期，必须 fail-fast。
        raise TypeError("target_fpr must be number")
    if target_fpr <= 0:
        raise ValueError("target_fpr must be positive")

    raw = f"{float(target_fpr):.0e}"
    if "e" not in raw:
        return raw
    base, exp = raw.split("e", 1)
    sign = exp[0]
    digits = exp[1:]
    if len(digits) == 1:
        digits = f"0{digits}"
    return f"{base}e{sign}{digits}"


def validate_thresholds_metadata(metadata: Dict[str, Any]) -> None:
    """
    功能：校验 threshold metadata 的稳定 schema。

    Validate threshold metadata schema and types.

    Args:
        metadata: Metadata mapping.

    Returns:
        None.

    Raises:
        TypeError: If metadata is invalid.
        ValueError: If schema or types are invalid.
    """
    if not isinstance(metadata, dict):
        # metadata 类型不符合预期，必须 fail-fast。
        raise TypeError("metadata must be dict")

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"metadata missing required keys: {missing}")

    for key in ["method", "null_source", "calibration_date", "quantile_method"]:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"metadata field must be non-empty str: field_path=metadata.{key}")

    n_null = metadata.get("n_null")
    if not isinstance(n_null, int) or n_null < 0:
        raise ValueError("metadata.n_null must be non-negative int")

    target_fprs = metadata.get("target_fprs")
    if not isinstance(target_fprs, list) or not target_fprs:
        raise ValueError("metadata.target_fprs must be non-empty list")
    for idx, value in enumerate(target_fprs):
        if not isinstance(value, (int, float)):
            raise ValueError(f"metadata.target_fprs[{idx}] must be number")


def load_thresholds_canonical(thresholds_path: str, target_fpr: float) -> Dict[str, Any]:
    """
    功能：按冻结口径加载阈值。

    Load thresholds from JSON and return canonical selection output.

    Args:
        thresholds_path: Path to thresholds JSON file.
        target_fpr: Target false positive rate.

    Returns:
        Threshold info mapping.

    Raises:
        TypeError: If inputs are invalid.
        ValueError: If file or data is invalid.
    """
    if not isinstance(thresholds_path, str) or not thresholds_path:
        # thresholds_path 类型不符合预期，必须 fail-fast。
        raise TypeError("thresholds_path must be non-empty str")
    if not isinstance(target_fpr, (int, float)):
        # target_fpr 类型不符合预期，必须 fail-fast。
        raise TypeError("target_fpr must be number")

    path = Path(thresholds_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"thresholds_path not found: {thresholds_path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            thresholds = json.load(f)
    except Exception as exc:
        raise ValueError(f"failed to load thresholds JSON: {exc}") from exc
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds JSON root must be dict")

    return select_thresholds_np(thresholds, float(target_fpr))


def _extract_target_fpr(cfg: Dict[str, Any]) -> float:
    """
    功能：从 cfg 中提取 target_fpr。
    
    Extract target_fpr from config with fallback logic.
    
    Args:
        cfg: Configuration mapping.
    
    Returns:
        Extracted target_fpr value, or default 1e-6 if not found.
    """
    evaluate_section = cfg.get("evaluate")
    if isinstance(evaluate_section, dict) and "target_fpr" in evaluate_section:
        target_fpr = evaluate_section.get("target_fpr")
    else:
        thresholds_section = cfg.get("thresholds")
        if isinstance(thresholds_section, dict) and "target_fpr" in thresholds_section:
            target_fpr = thresholds_section.get("target_fpr")
        else:
            target_fpr = cfg.get("target_fpr")

    if target_fpr is None:
        raise ValueError("target_fpr must be provided")
    if not isinstance(target_fpr, (int, float)):
        raise ValueError("target_fpr must be number")
    return float(target_fpr)


def _lookup_threshold_value(thresholds: Dict[str, Any], fpr_key: str) -> float:
    """
    功能：从阈值表中提取阈值基线值。
    
    Lookup threshold value from thresholds mapping with fallback logic.
    
    Args:
        thresholds: Thresholds mapping loaded from JSON.
        fpr_key: Canonical FPR key string.
        
    Returns:
        Threshold value for the given FPR key, or 0.0 if not found.
    """
    value = None
    values_section = thresholds.get("values")
    if isinstance(values_section, dict):
        value = values_section.get(fpr_key)
    if value is None and fpr_key in thresholds:
        value = thresholds.get(fpr_key)
    if value is None:
        raise ValueError("threshold_value_used missing for fpr_key")
    if not isinstance(value, (int, float)):
        raise ValueError("threshold_value_used must be number")
    return float(value)


def compute_np_threshold_from_scores(
    scores: List[float],
    target_fpr: float,
    quantile_rule: str = "higher"
) -> tuple[float, Dict[str, Any]]:
    """
    功能：按 order-statistics 计算 NP 阈值。 

    Compute Neyman-Pearson threshold from null scores using order statistics.

    Args:
        scores: Null distribution score list.
        target_fpr: Target false positive rate in (0, 1).
        quantile_rule: Quantile selection rule, currently supports "higher".

    Returns:
        Tuple of (threshold_value, order_stat_info).

    Raises:
        TypeError: If input types are invalid.
        ValueError: If inputs are out of valid range.
    """
    if not isinstance(scores, list) or len(scores) == 0:
        raise ValueError("scores must be non-empty list")
    if not isinstance(target_fpr, (int, float)):
        raise TypeError("target_fpr must be number")
    if not isinstance(quantile_rule, str) or not quantile_rule:
        raise TypeError("quantile_rule must be non-empty str")
    if quantile_rule != "higher":
        raise ValueError("quantile_rule must be 'higher'")

    target_fpr_value = float(target_fpr)
    if target_fpr_value <= 0.0 or target_fpr_value >= 1.0:
        raise ValueError("target_fpr must be in (0, 1)")

    normalized_scores: List[float] = []
    for idx, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"scores[{idx}] must be number")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"scores[{idx}] must be finite")
        normalized_scores.append(value)

    sorted_scores = sorted(normalized_scores)
    n_samples = len(sorted_scores)

    # higher quantile: index = ceil(q*n) - 1, q = 1-target_fpr
    quantile = 1.0 - target_fpr_value
    raw_rank = int(math.ceil(quantile * n_samples))
    rank_1based = min(max(1, raw_rank), n_samples)
    index_0based = rank_1based - 1
    threshold_value = float(sorted_scores[index_0based])

    order_stat_info = {
        "n_samples": n_samples,
        "target_fpr": target_fpr_value,
        "quantile": quantile,
        "quantile_rule": "higher",
        "order_stat_rank_1based": rank_1based,
        "order_stat_index_0based": index_0based,
        "ties_policy": "higher",
    }
    return threshold_value, order_stat_info
