"""
File purpose: Shared PW02/PW04 image quality metric helpers for append-only paper exports.
Module type: Semi-general module
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

import numpy as np
from PIL import Image

from main.evaluation.image_quality import compute_psnr, compute_ssim
from scripts.notebook_runtime_common import normalize_path_value


_LPIPS_MODELS: Dict[str, Any] = {}
_LPIPS_MODEL_ERRORS: Dict[str, str] = {}
_CLIP_MODELS: Dict[str, Any] = {}
_CLIP_PREPROCESSES: Dict[str, Any] = {}
_CLIP_TOKENIZERS: Dict[str, Any] = {}
_CLIP_MODEL_ERRORS: Dict[str, str] = {}
CLIP_UNAVAILABLE_REASON = "requires frozen quality model identity and bootstrap contract"
CLIP_MODEL_NAME = "open_clip:ViT-B-32/laion2b_s34b_b79k"
_CLIP_MODEL_ARCH = "ViT-B-32"
_CLIP_MODEL_PRETRAINED = "laion2b_s34b_b79k"
QUALITY_TORCH_DEVICE_ENV = "PW_QUALITY_TORCH_DEVICE"
QUALITY_LPIPS_BATCH_SIZE_ENV = "PW_QUALITY_LPIPS_BATCH_SIZE"
QUALITY_CLIP_BATCH_SIZE_ENV = "PW_QUALITY_CLIP_BATCH_SIZE"
DEFAULT_QUALITY_TORCH_DEVICE = "cpu"
DEFAULT_QUALITY_BATCH_SIZE = 1


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


def _normalize_batch_size(batch_size_value: Any, label: str) -> int:
    """
    功能：规范化 quality metric batch size 输入。

    Normalize one quality-metric batch size value.

    Args:
        batch_size_value: Candidate batch size.
        label: Human-readable label.

    Returns:
        Positive batch size.

    Raises:
        TypeError: If the batch size is invalid.
    """
    try:
        normalized_batch_size = int(batch_size_value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be positive int") from exc
    if normalized_batch_size <= 0:
        raise TypeError(f"{label} must be positive int")
    return normalized_batch_size


def _resolve_quality_runtime_options(
    *,
    torch_device: str | None,
    lpips_batch_size: int | None,
    clip_batch_size: int | None,
) -> Dict[str, Any]:
    """
    功能：解析 quality metric 的 device 与 batch 运行参数。

    Resolve device and batch runtime options for quality metrics.

    Args:
        torch_device: Optional explicit torch device string.
        lpips_batch_size: Optional explicit LPIPS batch size.
        clip_batch_size: Optional explicit CLIP batch size.

    Returns:
        Runtime option mapping.

    Raises:
        TypeError: If any runtime option is invalid.
    """
    resolved_torch_device = (
        torch_device
        if isinstance(torch_device, str) and torch_device.strip()
        else os.environ.get(QUALITY_TORCH_DEVICE_ENV, DEFAULT_QUALITY_TORCH_DEVICE)
    )
    if not isinstance(resolved_torch_device, str) or not resolved_torch_device.strip():
        raise TypeError("torch_device must be non-empty str when provided")

    lpips_batch_size_raw = (
        lpips_batch_size
        if lpips_batch_size is not None
        else os.environ.get(QUALITY_LPIPS_BATCH_SIZE_ENV, DEFAULT_QUALITY_BATCH_SIZE)
    )
    clip_batch_size_raw = (
        clip_batch_size
        if clip_batch_size is not None
        else os.environ.get(QUALITY_CLIP_BATCH_SIZE_ENV, DEFAULT_QUALITY_BATCH_SIZE)
    )
    return {
        "torch_device": resolved_torch_device.strip(),
        "lpips_batch_size": _normalize_batch_size(lpips_batch_size_raw, "lpips_batch_size"),
        "clip_batch_size": _normalize_batch_size(clip_batch_size_raw, "clip_batch_size"),
    }


def _get_lpips_model(
    torch_device: str = DEFAULT_QUALITY_TORCH_DEVICE,
) -> tuple[Any | None, str | None]:
    """
    Load and cache the LPIPS model for image-pair quality evaluation.

    Args:
        None.

    Returns:
        Tuple of (model_or_none, error_reason_or_none).
    """
    resolved_torch_device = str(torch_device).strip()
    if not resolved_torch_device:
        raise TypeError("torch_device must be non-empty str")

    if resolved_torch_device in _LPIPS_MODELS:
        return _LPIPS_MODELS[resolved_torch_device], None
    if resolved_torch_device in _LPIPS_MODEL_ERRORS:
        return None, _LPIPS_MODEL_ERRORS[resolved_torch_device]

    try:
        import lpips  # type: ignore
        import torch
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        _LPIPS_MODEL_ERRORS[resolved_torch_device] = error_message
        return None, error_message

    try:
        model = lpips.LPIPS(net="alex")
        model.eval()
        model.to(torch.device(resolved_torch_device))
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        _LPIPS_MODEL_ERRORS[resolved_torch_device] = error_message
        return None, error_message

    _LPIPS_MODELS[resolved_torch_device] = model
    return model, None


def _compute_lpips_value(
    reference_image: np.ndarray[Any, Any],
    candidate_image: np.ndarray[Any, Any],
    torch_device: str = DEFAULT_QUALITY_TORCH_DEVICE,
) -> float:
    """
    Compute LPIPS perceptual distance for one RGB image pair.

    Args:
        reference_image: Reference RGB image array.
        candidate_image: Candidate RGB image array.

    Returns:
        LPIPS distance where smaller indicates higher perceptual similarity.

    Raises:
        RuntimeError: If the LPIPS model is unavailable.
    """
    model, model_error = _get_lpips_model(torch_device=torch_device)
    if model is None:
        raise RuntimeError(model_error or "LPIPS model unavailable")

    import torch

    device = torch.device(torch_device)
    reference_tensor = torch.from_numpy(reference_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    candidate_tensor = torch.from_numpy(candidate_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    reference_tensor = (reference_tensor / 127.5) - 1.0
    candidate_tensor = (candidate_tensor / 127.5) - 1.0
    with torch.no_grad():
        lpips_value = model(reference_tensor, candidate_tensor)
    return float(lpips_value.reshape(-1)[0].item())


def _compute_lpips_values_batch(
    reference_images: Sequence[np.ndarray[Any, Any]],
    candidate_images: Sequence[np.ndarray[Any, Any]],
    torch_device: str = DEFAULT_QUALITY_TORCH_DEVICE,
) -> List[float]:
    """
    Compute LPIPS perceptual distance for one aligned image batch.

    Args:
        reference_images: Reference RGB image arrays.
        candidate_images: Candidate RGB image arrays.
        torch_device: Torch device used for model and tensors.

    Returns:
        LPIPS distances aligned with the input order.

    Raises:
        RuntimeError: If the LPIPS model is unavailable.
        ValueError: If the batch inputs are inconsistent.
    """
    if len(reference_images) != len(candidate_images):
        raise ValueError("reference_images and candidate_images must have the same length")
    if not reference_images:
        return []

    model, model_error = _get_lpips_model(torch_device=torch_device)
    if model is None:
        raise RuntimeError(model_error or "LPIPS model unavailable")

    import torch

    device = torch.device(torch_device)
    reference_tensor = torch.stack(
        [torch.from_numpy(image).permute(2, 0, 1).float() for image in reference_images],
        dim=0,
    ).to(device)
    candidate_tensor = torch.stack(
        [torch.from_numpy(image).permute(2, 0, 1).float() for image in candidate_images],
        dim=0,
    ).to(device)
    reference_tensor = (reference_tensor / 127.5) - 1.0
    candidate_tensor = (candidate_tensor / 127.5) - 1.0
    with torch.no_grad():
        lpips_values = model(reference_tensor, candidate_tensor)
    return [float(value.item()) for value in lpips_values.reshape(-1)]


def _get_clip_model_components(
    torch_device: str = DEFAULT_QUALITY_TORCH_DEVICE,
) -> tuple[Any | None, Any | None, Any | None, str | None]:
    """
    Load and cache the frozen CLIP model components for image-text similarity.

    Args:
        None.

    Returns:
        Tuple of (model_or_none, preprocess_or_none, tokenizer_or_none, error_reason_or_none).
    """
    resolved_torch_device = str(torch_device).strip()
    if not resolved_torch_device:
        raise TypeError("torch_device must be non-empty str")

    if (
        resolved_torch_device in _CLIP_MODELS
        and resolved_torch_device in _CLIP_PREPROCESSES
        and resolved_torch_device in _CLIP_TOKENIZERS
    ):
        return (
            _CLIP_MODELS[resolved_torch_device],
            _CLIP_PREPROCESSES[resolved_torch_device],
            _CLIP_TOKENIZERS[resolved_torch_device],
            None,
        )
    if resolved_torch_device in _CLIP_MODEL_ERRORS:
        return None, None, None, _CLIP_MODEL_ERRORS[resolved_torch_device]

    try:
        import open_clip  # type: ignore
        import torch
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        _CLIP_MODEL_ERRORS[resolved_torch_device] = error_message
        return None, None, None, error_message

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL_ARCH,
            pretrained=_CLIP_MODEL_PRETRAINED,
            device=torch.device(resolved_torch_device),
        )
        tokenizer = open_clip.get_tokenizer(_CLIP_MODEL_ARCH)
        model.eval()
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        _CLIP_MODEL_ERRORS[resolved_torch_device] = error_message
        return None, None, None, error_message

    _CLIP_MODELS[resolved_torch_device] = model
    _CLIP_PREPROCESSES[resolved_torch_device] = preprocess
    _CLIP_TOKENIZERS[resolved_torch_device] = tokenizer
    return model, preprocess, tokenizer, None


def _compute_clip_text_similarity(
    candidate_image: np.ndarray[Any, Any],
    prompt_text: str,
    torch_device: str = DEFAULT_QUALITY_TORCH_DEVICE,
) -> float:
    """
    Compute CLIP image-text cosine similarity for one candidate image and prompt.

    Args:
        candidate_image: Candidate RGB image array.
        prompt_text: Prompt text paired with the candidate image.

    Returns:
        CLIP cosine similarity where larger indicates stronger semantic alignment.

    Raises:
        RuntimeError: If the CLIP model is unavailable.
        ValueError: If prompt_text is invalid.
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt_text must be non-empty str")

    model, preprocess, tokenizer, model_error = _get_clip_model_components(torch_device=torch_device)
    if model is None or preprocess is None or tokenizer is None:
        raise RuntimeError(model_error or "CLIP model unavailable")

    import torch

    device = torch.device(torch_device)
    image_tensor = preprocess(Image.fromarray(candidate_image)).unsqueeze(0).to(device)
    text_tensor = tokenizer([prompt_text.strip()])
    if hasattr(text_tensor, "to"):
        text_tensor = text_tensor.to(device)
    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        similarity_value = torch.matmul(image_features, text_features.T)
    return float(similarity_value.reshape(-1)[0].item())


def _compute_clip_text_similarity_batch(
    candidate_images: Sequence[np.ndarray[Any, Any]],
    prompt_texts: Sequence[str],
    torch_device: str = DEFAULT_QUALITY_TORCH_DEVICE,
) -> List[float]:
    """
    Compute CLIP image-text cosine similarity for one aligned batch.

    Args:
        candidate_images: Candidate RGB image arrays.
        prompt_texts: Prompt texts aligned with candidate_images.
        torch_device: Torch device used for model and tensors.

    Returns:
        Cosine similarities aligned with the input order.

    Raises:
        RuntimeError: If the CLIP model is unavailable.
        ValueError: If the prompts are invalid or the batch lengths mismatch.
    """
    if len(candidate_images) != len(prompt_texts):
        raise ValueError("candidate_images and prompt_texts must have the same length")
    if not candidate_images:
        return []

    cleaned_prompts: List[str] = []
    for prompt_text in prompt_texts:
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("prompt_texts items must be non-empty str")
        cleaned_prompts.append(prompt_text.strip())

    model, preprocess, tokenizer, model_error = _get_clip_model_components(torch_device=torch_device)
    if model is None or preprocess is None or tokenizer is None:
        raise RuntimeError(model_error or "CLIP model unavailable")

    import torch

    device = torch.device(torch_device)
    image_tensor = torch.stack(
        [preprocess(Image.fromarray(candidate_image)) for candidate_image in candidate_images],
        dim=0,
    ).to(device)
    text_tensor = tokenizer(cleaned_prompts)
    if hasattr(text_tensor, "to"):
        text_tensor = text_tensor.to(device)
    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        similarity_values = (image_features * text_features).sum(dim=-1)
    return [float(value.item()) for value in similarity_values.reshape(-1)]


def _call_lpips_value_compat(
    reference_image: np.ndarray[Any, Any],
    candidate_image: np.ndarray[Any, Any],
    *,
    torch_device: str,
) -> float:
    """
    功能：兼容旧 monkeypatch 签名调用单样本 LPIPS 计算。

    Call the single-item LPIPS helper while remaining compatible with legacy monkeypatches.

    Args:
        reference_image: Reference RGB image array.
        candidate_image: Candidate RGB image array.
        torch_device: Torch device string.

    Returns:
        LPIPS distance for one image pair.
    """
    try:
        return _compute_lpips_value(
            reference_image,
            candidate_image,
            torch_device=torch_device,
        )
    except TypeError as exc:
        if "torch_device" not in str(exc):
            raise
    return _compute_lpips_value(reference_image, candidate_image)


def _call_clip_text_similarity_compat(
    candidate_image: np.ndarray[Any, Any],
    prompt_text: str,
    *,
    torch_device: str,
) -> float:
    """
    功能：兼容旧 monkeypatch 签名调用单样本 CLIP 计算。

    Call the single-item CLIP helper while remaining compatible with legacy monkeypatches.

    Args:
        candidate_image: Candidate RGB image array.
        prompt_text: Prompt text.
        torch_device: Torch device string.

    Returns:
        CLIP similarity for one image-text pair.
    """
    try:
        return _compute_clip_text_similarity(
            candidate_image,
            prompt_text,
            torch_device=torch_device,
        )
    except TypeError as exc:
        if "torch_device" not in str(exc):
            raise
    return _compute_clip_text_similarity(candidate_image, prompt_text)


def build_quality_metrics_from_pairs(
    *,
    pair_specs: Sequence[Mapping[str, Any]],
    reference_path_key: str,
    candidate_path_key: str,
    pair_id_key: str,
    text_key: str | None = None,
    extra_metadata_keys: Sequence[str] | None = None,
    torch_device: str | None = None,
    lpips_batch_size: int | None = None,
    clip_batch_size: int | None = None,
) -> Dict[str, Any]:
    """
    功能：对一组图像路径对计算质量指标摘要。

    Compute image-pair quality and optional CLIP text-alignment summaries.

    Args:
        pair_specs: Pair specifications.
        reference_path_key: Key containing the reference image path.
        candidate_path_key: Key containing the candidate image path.
        pair_id_key: Key containing the stable pair identifier.
        text_key: Optional key containing the prompt text for CLIP alignment.
        extra_metadata_keys: Additional metadata keys copied into each pair row.
        torch_device: Optional torch device string used by LPIPS and CLIP.
        lpips_batch_size: Optional LPIPS batch size.
        clip_batch_size: Optional CLIP batch size.

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
    if text_key is not None and (not isinstance(text_key, str) or not text_key):
        raise TypeError("text_key must be non-empty str when provided")
    metadata_keys = list(extra_metadata_keys) if extra_metadata_keys is not None else []
    runtime_options = _resolve_quality_runtime_options(
        torch_device=torch_device,
        lpips_batch_size=lpips_batch_size,
        clip_batch_size=clip_batch_size,
    )
    resolved_torch_device = str(runtime_options["torch_device"])
    resolved_lpips_batch_size = int(runtime_options["lpips_batch_size"])
    resolved_clip_batch_size = int(runtime_options["clip_batch_size"])

    pair_rows: List[Dict[str, Any]] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []
    clip_values: List[float] = []
    lpips_pending: List[tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    clip_pending: List[tuple[int, np.ndarray[Any, Any], str]] = []
    missing_count = 0
    error_count = 0
    lpips_reason: str | None = None
    clip_reason: str | None = None
    clip_missing_text_count = 0
    clip_error_count = 0

    for pair_spec in pair_specs:
        if not isinstance(pair_spec, Mapping):
            raise TypeError("pair_specs items must be Mapping")

        pair_id = pair_spec.get(pair_id_key)
        reference_value = pair_spec.get(reference_path_key)
        candidate_value = pair_spec.get(candidate_path_key)
        prompt_text = pair_spec.get(text_key) if text_key is not None else None
        pair_row: Dict[str, Any] = {
            "pair_id": pair_id,
            "reference_image_path": reference_value,
            "candidate_image_path": candidate_value,
            "status": "missing_binding",
            "failure_reason": None,
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "clip_text_similarity": None,
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
        pair_row_index = len(pair_rows)
        if resolved_lpips_batch_size > 1:
            lpips_pending.append((pair_row_index, reference_image, candidate_image))
        else:
            try:
                lpips_value = _call_lpips_value_compat(
                    reference_image,
                    candidate_image,
                    torch_device=resolved_torch_device,
                )
            except Exception as exc:
                if lpips_reason is None:
                    lpips_reason = f"{type(exc).__name__}: {exc}"
            else:
                pair_row["lpips"] = lpips_value
                lpips_values.append(lpips_value)

        if text_key is not None:
            if isinstance(prompt_text, str) and prompt_text.strip():
                if resolved_clip_batch_size > 1:
                    clip_pending.append((pair_row_index, candidate_image, prompt_text.strip()))
                else:
                    try:
                        clip_value = _call_clip_text_similarity_compat(
                            candidate_image,
                            prompt_text,
                            torch_device=resolved_torch_device,
                        )
                    except Exception as exc:
                        clip_error_count += 1
                        if clip_reason is None:
                            clip_reason = f"{type(exc).__name__}: {exc}"
                    else:
                        pair_row["clip_text_similarity"] = clip_value
                        clip_values.append(clip_value)
            else:
                clip_missing_text_count += 1

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        pair_rows.append(pair_row)

    if resolved_lpips_batch_size > 1 and lpips_pending:
        for batch_start in range(0, len(lpips_pending), resolved_lpips_batch_size):
            batch = lpips_pending[batch_start : batch_start + resolved_lpips_batch_size]
            try:
                batch_values = _compute_lpips_values_batch(
                    [reference_image for _, reference_image, _ in batch],
                    [candidate_image for _, _, candidate_image in batch],
                    torch_device=resolved_torch_device,
                )
            except Exception as exc:
                if lpips_reason is None:
                    lpips_reason = f"{type(exc).__name__}: {exc}"
                for pair_row_index, reference_image, candidate_image in batch:
                    try:
                        lpips_value = _call_lpips_value_compat(
                            reference_image,
                            candidate_image,
                            torch_device=resolved_torch_device,
                        )
                    except Exception as single_exc:
                        if lpips_reason is None:
                            lpips_reason = f"{type(single_exc).__name__}: {single_exc}"
                    else:
                        pair_rows[pair_row_index]["lpips"] = lpips_value
                        lpips_values.append(lpips_value)
            else:
                for (pair_row_index, _, _), lpips_value in zip(batch, batch_values, strict=False):
                    pair_rows[pair_row_index]["lpips"] = lpips_value
                    lpips_values.append(lpips_value)

    if resolved_clip_batch_size > 1 and clip_pending:
        for batch_start in range(0, len(clip_pending), resolved_clip_batch_size):
            batch = clip_pending[batch_start : batch_start + resolved_clip_batch_size]
            try:
                batch_values = _compute_clip_text_similarity_batch(
                    [candidate_image for _, candidate_image, _ in batch],
                    [prompt_text for _, _, prompt_text in batch],
                    torch_device=resolved_torch_device,
                )
            except Exception as exc:
                clip_error_count += len(batch)
                if clip_reason is None:
                    clip_reason = f"{type(exc).__name__}: {exc}"
                for pair_row_index, candidate_image, prompt_text in batch:
                    try:
                        clip_value = _call_clip_text_similarity_compat(
                            candidate_image,
                            prompt_text,
                            torch_device=resolved_torch_device,
                        )
                    except Exception as single_exc:
                        if clip_reason is None:
                            clip_reason = f"{type(single_exc).__name__}: {single_exc}"
                    else:
                        clip_error_count -= 1
                        pair_rows[pair_row_index]["clip_text_similarity"] = clip_value
                        clip_values.append(clip_value)
            else:
                for (pair_row_index, _, _), clip_value in zip(batch, batch_values, strict=False):
                    pair_rows[pair_row_index]["clip_text_similarity"] = clip_value
                    clip_values.append(clip_value)

    successful_count = len(psnr_values)
    mean_psnr = float(np.mean(psnr_values)) if psnr_values else None
    mean_ssim = float(np.mean(ssim_values)) if ssim_values else None
    mean_lpips = float(np.mean(lpips_values)) if lpips_values else None
    status = "ok" if successful_count > 0 else "unavailable"
    availability_reason = None if successful_count > 0 else "no_valid_image_pairs"
    lpips_status = "ok" if lpips_values else "not_available"
    clip_sample_count = len(clip_values)
    mean_clip_text_similarity = float(np.mean(clip_values)) if clip_values else None
    if text_key is None:
        clip_status = "not_available"
        clip_reason = "prompt text key not configured for CLIP quality metric"
        clip_model_name: str | None = None
    else:
        clip_model_name = CLIP_MODEL_NAME
        if clip_sample_count > 0 and clip_error_count == 0 and clip_missing_text_count == 0:
            clip_status = "ok"
            clip_reason = None
        elif clip_sample_count > 0:
            clip_status = "partial"
            partial_reasons: List[str] = []
            if clip_missing_text_count > 0:
                partial_reasons.append(f"prompt text unavailable for {clip_missing_text_count} valid image pairs")
            if clip_error_count > 0:
                partial_reasons.append(f"CLIP computation failed for {clip_error_count} valid image pairs")
            clip_reason = "; ".join(partial_reasons) if partial_reasons else clip_reason
        elif successful_count <= 0:
            clip_status = "not_available"
            clip_reason = "no_valid_image_pairs"
        elif clip_missing_text_count > 0 and clip_error_count == 0:
            clip_status = "not_available"
            clip_reason = f"prompt text unavailable for all {clip_missing_text_count} valid image pairs"
        else:
            clip_status = "not_available"
            clip_reason = clip_reason or "CLIP model unavailable"

    prompt_text_expected = text_key is not None
    prompt_text_available_count = (
        max(successful_count - clip_missing_text_count, 0)
        if prompt_text_expected
        else None
    )
    if not prompt_text_expected:
        prompt_text_coverage_status = "not_configured"
        prompt_text_coverage_reason = "prompt text key not configured for CLIP quality metric"
    elif successful_count <= 0:
        prompt_text_coverage_status = "not_available"
        prompt_text_coverage_reason = "no_valid_image_pairs"
    elif clip_missing_text_count <= 0:
        prompt_text_coverage_status = "ok"
        prompt_text_coverage_reason = None
    elif clip_missing_text_count < successful_count:
        prompt_text_coverage_status = "partial"
        prompt_text_coverage_reason = (
            f"prompt text unavailable for {clip_missing_text_count}/{successful_count} valid image pairs"
        )
    else:
        prompt_text_coverage_status = "not_available"
        prompt_text_coverage_reason = (
            f"prompt text unavailable for all {clip_missing_text_count} valid image pairs"
        )

    quality_readiness_reasons: List[str] = []
    if successful_count <= 0:
        quality_readiness_status = "not_ready"
        quality_readiness_reason = "no_valid_image_pairs"
    else:
        if successful_count != len(pair_specs):
            quality_readiness_reasons.append(
                f"valid image pairs available for {successful_count}/{len(pair_specs)} expected bindings"
            )
        if lpips_status != "ok":
            quality_readiness_reasons.append(lpips_reason or "LPIPS model unavailable")
        if prompt_text_expected and prompt_text_coverage_status not in {"ok", "not_configured"}:
            quality_readiness_reasons.append(
                prompt_text_coverage_reason or "prompt text coverage incomplete"
            )
        if prompt_text_expected and clip_status != "ok":
            quality_readiness_reasons.append(clip_reason or CLIP_UNAVAILABLE_REASON)
        if quality_readiness_reasons:
            quality_readiness_status = "partial"
            quality_readiness_reason = "; ".join(dict.fromkeys(quality_readiness_reasons))
        else:
            quality_readiness_status = "ready"
            quality_readiness_reason = None

    return {
        "status": status,
        "availability_reason": availability_reason,
        "expected_count": len(pair_specs),
        "count": successful_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "lpips_status": lpips_status,
        "lpips_reason": None if lpips_status == "ok" else (lpips_reason or "LPIPS model unavailable"),
        "mean_lpips": mean_lpips,
        "mean_clip_text_similarity": mean_clip_text_similarity,
        "clip_model_name": clip_model_name,
        "clip_sample_count": clip_sample_count,
        "clip_status": clip_status,
        "clip_reason": None if clip_status == "ok" else (clip_reason or CLIP_UNAVAILABLE_REASON),
        "quality_runtime": {
            "torch_device": resolved_torch_device,
            "lpips_batch_size": resolved_lpips_batch_size,
            "clip_batch_size": resolved_clip_batch_size,
        },
        "prompt_text_expected": prompt_text_expected,
        "prompt_text_available_count": prompt_text_available_count,
        "prompt_text_missing_count": clip_missing_text_count if prompt_text_expected else None,
        "prompt_text_coverage_status": prompt_text_coverage_status,
        "prompt_text_coverage_reason": prompt_text_coverage_reason,
        "quality_readiness_status": quality_readiness_status,
        "quality_readiness_reason": quality_readiness_reason,
        "quality_readiness_blocking": quality_readiness_status != "ready",
        "quality_readiness_required_for_formal_release": True,
        "pair_rows": pair_rows,
    }