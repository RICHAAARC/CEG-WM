"""
File purpose: Shared PW02/PW04 image quality metric helpers for append-only paper exports.
Module type: Semi-general module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

import numpy as np
from PIL import Image

from main.evaluation.image_quality import compute_psnr, compute_ssim
from scripts.notebook_runtime_common import normalize_path_value


def _load_required_json_dict(path_obj: Path, label: str) -> Dict[str, Any]:
    """
    功能：读取必需的 JSON 对象文件。

    Load one required JSON object file.

    Args:
        path_obj: JSON file path.
        label: Human-readable label.

    Returns:
        Parsed JSON mapping.

    Raises:
        TypeError: If the input types are invalid.
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON root is not an object.
    """
    if not isinstance(path_obj, Path):
        raise TypeError("path_obj must be Path")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {normalize_path_value(path_obj)}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be JSON object: {normalize_path_value(path_obj)}")
    return cast(Dict[str, Any], payload)


def resolve_optional_artifact_path(view_payload: Mapping[str, Any], label: str) -> Path | None:
    """
    功能：从 optional artifact view 中解析可用路径。

    Resolve the staged path from an optional artifact view.

    Args:
        view_payload: Optional artifact view payload.
        label: Human-readable label.

    Returns:
        Resolved path when the artifact exists, otherwise None.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(view_payload, Mapping):
        raise TypeError(f"{label} must be Mapping")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be non-empty str")

    exists_value = view_payload.get("exists")
    if exists_value is not True:
        return None

    path_value = view_payload.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    return Path(path_value).expanduser().resolve()


def resolve_preview_persisted_artifact_path(preview_record_view: Mapping[str, Any]) -> Path | None:
    """
    功能：从 preview_generation_record view 解析实际输出图像路径。

    Resolve the persisted preview artifact path from a preview-generation record view.

    Args:
        preview_record_view: Preview-generation record artifact view.

    Returns:
        Resolved persisted artifact path when available, otherwise None.
    """
    preview_record_path = resolve_optional_artifact_path(preview_record_view, "preview_record_view")
    if preview_record_path is None:
        return None

    preview_record = _load_required_json_dict(preview_record_path, "preview generation record")
    persisted_artifact_path = preview_record.get("persisted_artifact_path")
    if not isinstance(persisted_artifact_path, str) or not persisted_artifact_path.strip():
        return None
    return Path(persisted_artifact_path).expanduser().resolve()


def _load_rgb_image(image_path: Path) -> np.ndarray[Any, Any]:
    """
    功能：读取 RGB 图像并返回 numpy 数组。

    Load one RGB image into a numpy array.

    Args:
        image_path: Image file path.

    Returns:
        RGB image array.

    Raises:
        TypeError: If image_path is invalid.
        FileNotFoundError: If the image file does not exist.
    """
    if not isinstance(image_path, Path):
        raise TypeError("image_path must be Path")
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"image file not found: {normalize_path_value(image_path)}")
    with Image.open(image_path) as image_obj:
        return np.asarray(image_obj.convert("RGB"))


def build_quality_metrics_from_pairs(
    *,
    pair_specs: Sequence[Mapping[str, Any]],
    reference_path_key: str,
    candidate_path_key: str,
    pair_id_key: str,
    extra_metadata_keys: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    功能：对一组图像路径对计算质量指标摘要。

    Compute PSNR and SSIM summaries for a sequence of image-path pairs.

    Args:
        pair_specs: Pair specifications.
        reference_path_key: Key containing the reference image path.
        candidate_path_key: Key containing the candidate image path.
        pair_id_key: Key containing the stable pair identifier.
        extra_metadata_keys: Additional metadata keys copied into each pair row.

    Returns:
        Quality summary payload with aggregate metrics and per-pair rows.

    Raises:
        TypeError: If inputs are invalid.
    """
    if not isinstance(pair_specs, Sequence):
        raise TypeError("pair_specs must be Sequence")
    if not isinstance(reference_path_key, str) or not reference_path_key:
        raise TypeError("reference_path_key must be non-empty str")
    if not isinstance(candidate_path_key, str) or not candidate_path_key:
        raise TypeError("candidate_path_key must be non-empty str")
    if not isinstance(pair_id_key, str) or not pair_id_key:
        raise TypeError("pair_id_key must be non-empty str")
    metadata_keys = list(extra_metadata_keys) if extra_metadata_keys is not None else []

    pair_rows: List[Dict[str, Any]] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    missing_count = 0
    error_count = 0

    for pair_spec in pair_specs:
        if not isinstance(pair_spec, Mapping):
            raise TypeError("pair_specs items must be Mapping")

        pair_id = pair_spec.get(pair_id_key)
        reference_value = pair_spec.get(reference_path_key)
        candidate_value = pair_spec.get(candidate_path_key)
        pair_row: Dict[str, Any] = {
            "pair_id": pair_id,
            "reference_image_path": reference_value,
            "candidate_image_path": candidate_value,
            "status": "missing_binding",
            "failure_reason": None,
            "psnr": None,
            "ssim": None,
        }
        for metadata_key in metadata_keys:
            pair_row[metadata_key] = pair_spec.get(metadata_key)

        if not isinstance(pair_id, str) or not pair_id:
            pair_row["failure_reason"] = f"missing_{pair_id_key}"
            missing_count += 1
            pair_rows.append(pair_row)
            continue

        if not isinstance(reference_value, str) or not reference_value.strip():
            pair_row["status"] = "missing_reference"
            pair_row["failure_reason"] = f"missing_{reference_path_key}"
            missing_count += 1
            pair_rows.append(pair_row)
            continue

        if not isinstance(candidate_value, str) or not candidate_value.strip():
            pair_row["status"] = "missing_candidate"
            pair_row["failure_reason"] = f"missing_{candidate_path_key}"
            missing_count += 1
            pair_rows.append(pair_row)
            continue

        reference_path = Path(reference_value).expanduser().resolve()
        candidate_path = Path(candidate_value).expanduser().resolve()
        pair_row["reference_image_path"] = normalize_path_value(reference_path)
        pair_row["candidate_image_path"] = normalize_path_value(candidate_path)
        try:
            reference_image = _load_rgb_image(reference_path)
            candidate_image = _load_rgb_image(candidate_path)
            psnr_value = float(compute_psnr(reference_image, candidate_image))
            ssim_value = float(compute_ssim(reference_image, candidate_image))
        except FileNotFoundError as exc:
            pair_row["status"] = "missing_file"
            pair_row["failure_reason"] = str(exc)
            missing_count += 1
            pair_rows.append(pair_row)
            continue
        except Exception as exc:
            pair_row["status"] = "error"
            pair_row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            error_count += 1
            pair_rows.append(pair_row)
            continue

        pair_row["status"] = "ok"
        pair_row["psnr"] = psnr_value
        pair_row["ssim"] = ssim_value
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        pair_rows.append(pair_row)

    successful_count = len(psnr_values)
    mean_psnr = float(np.mean(psnr_values)) if psnr_values else None
    mean_ssim = float(np.mean(ssim_values)) if ssim_values else None
    status = "ok" if successful_count > 0 else "unavailable"
    availability_reason = None if successful_count > 0 else "no_valid_image_pairs"

    return {
        "status": status,
        "availability_reason": availability_reason,
        "expected_count": len(pair_specs),
        "count": successful_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "pair_rows": pair_rows,
    }