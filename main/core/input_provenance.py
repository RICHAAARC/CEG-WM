"""
外部输入资产溯源

功能说明：
- 提供输入域可锚定的固定入口与产物形态。
- 缺失必须显式为 <absent>，禁止静默。
- v1 阶段不将其纳入强门禁 must_enforce，但必须"必产物、缺失显式化"。
"""

from __future__ import annotations

from typing import Any, Dict, cast

from main.core import digests


def build_input_provenance(cfg: Dict[str, Any], command: str) -> Dict[str, Any]:
    """
    功能：构造输入来源审计对象。

    Build input provenance audit object from config.
    Extract possible input fields; if not present, explicitly mark as <absent>.

    Args:
        cfg: Configuration mapping.
        command: Command name (embed/detect/calibrate/evaluate).

    Returns:
        Input provenance mapping with explicit <absent> semantics.

    Raises:
        TypeError: If inputs are invalid.
    """
    cfg_obj: Any = cfg
    if not isinstance(cfg_obj, dict):
        # cfg 类型不符合预期，必须 fail-fast。
        raise TypeError("cfg must be dict")
    command_obj: Any = command
    if not isinstance(command_obj, str) or not command_obj:
        # command 输入不合法，必须 fail-fast。
        raise TypeError("command must be non-empty str")

    cfg_mapping = cast(Dict[str, Any], cfg_obj)
    normalized_command = command_obj

    provenance = {
        "command": normalized_command,
        "prompt_sha256": "<absent>",
        "negative_prompt_sha256": "<absent>",
        "dataset_manifest_sha256": "<absent>",
        "image_list_sha256": "<absent>",
        "resolution": "<absent>",
        "num_steps": "<absent>"
    }

    # 提取 prompt 并计算 sha256。
    prompt = cfg_mapping.get("prompt")
    if isinstance(prompt, str) and prompt:
        provenance["prompt_sha256"] = digests.canonical_sha256(prompt)

    # 提取 negative_prompt 并计算 sha256。
    negative_prompt = cfg_mapping.get("negative_prompt")
    if isinstance(negative_prompt, str) and negative_prompt:
        provenance["negative_prompt_sha256"] = digests.canonical_sha256(negative_prompt)

    # 提取 dataset_manifest。
    dataset_manifest = cfg_mapping.get("dataset_manifest")
    if isinstance(dataset_manifest, str) and dataset_manifest:
        provenance["dataset_manifest_sha256"] = digests.canonical_sha256(dataset_manifest)

    # 提取 image_list。
    image_list = cfg_mapping.get("image_list")
    if isinstance(image_list, (list, str)) and image_list:
        provenance["image_list_sha256"] = digests.canonical_sha256(image_list)

    # 提取数值字段。
    resolution = cfg_mapping.get("resolution")
    if resolution is not None:
        provenance["resolution"] = str(resolution)

    num_steps = cfg_mapping.get("num_steps")
    if num_steps is not None:
        provenance["num_steps"] = str(num_steps)

    return provenance


def compute_input_provenance_digest(obj: Dict[str, Any]) -> str:
    """
    功能：计算输入来源审计对象的规范化 digest。

    Compute canonical digest for input provenance object.

    Args:
        obj: Input provenance mapping.

    Returns:
        Canonical SHA256 digest string.

    Raises:
        TypeError: If obj is invalid.
    """
    obj_value: Any = obj
    if not isinstance(obj_value, dict):
        # obj 类型不符合预期，必须 fail-fast。
        raise TypeError("obj must be dict")

    return digests.canonical_sha256(cast(Dict[str, Any], obj_value))
