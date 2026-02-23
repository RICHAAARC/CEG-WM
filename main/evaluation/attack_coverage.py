"""
File purpose: 攻击覆盖范围声明与摘要计算。
Module type: General module
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Dict

import numpy as np
from PIL import Image

from main.core import digests


def compute_attack_coverage_manifest() -> Dict[str, Any]:
    """
    功能：计算实现覆盖范围声明。

    Build deterministic attack coverage manifest for paper/runtime alignment.

    Args:
        None.

    Returns:
        Manifest mapping containing supported families, payload constraints,
        interpolation/boundary strategies, dependency versions, and digest.
    """
    manifest = {
        "manifest_version": "attack_coverage_v1",
        "supported_families": [
            "rotate",
            "resize",
            "crop",
            "translate",
            "jpeg_compression",
            "gaussian_noise",
            "gaussian_blur",
            "composite",
        ],
        "family_constraints": {
            "rotate": {
                "payload_support": {
                    "image": "any_degree_expand_false",
                    "latent": "multiples_of_90_only",
                },
                "interpolation": ["nearest", "bilinear", "bicubic", "lanczos"],
                "boundary": "expand_false",
            },
            "resize": {
                "payload_support": {
                    "image": "scale_resize_then_restore",
                    "latent": "nearest_index_resample_then_restore",
                },
                "interpolation": ["nearest", "bilinear", "bicubic", "lanczos"],
                "boundary": "shape_restored_to_original",
            },
            "crop": {
                "payload_support": {
                    "image": "center_crop_then_restore",
                    "latent": "center_crop_then_resample",
                },
                "interpolation": ["nearest", "bilinear", "bicubic", "lanczos"],
                "boundary": "center_policy",
            },
            "translate": {
                "payload_support": {
                    "image": "paste_with_zero_canvas",
                    "latent": "zero_fill_translation",
                },
                "boundary": "zero_fill",
            },
            "jpeg_compression": {
                "payload_support": {
                    "image": "supported",
                    "latent": "unsupported",
                },
                "boundary": "quality_clamped_1_100",
            },
            "gaussian_noise": {
                "payload_support": {
                    "image": "supported_via_array",
                    "latent": "supported",
                },
                "boundary": "dtype_restored",
            },
            "gaussian_blur": {
                "payload_support": {
                    "image": "pil_gaussian_blur",
                    "latent": "separable_convolution",
                },
                "boundary": "sigma_non_positive_identity",
            },
            "composite": {
                "payload_support": {
                    "image": "supported",
                    "latent": "supported_subject_to_each_step",
                },
                "boundary": "ordered_steps_seeded",
            },
        },
        "dependencies": {
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "torch": _read_optional_version("torch"),
        },
    }
    manifest["attack_coverage_digest"] = digests.canonical_sha256(manifest)
    return manifest


def _read_optional_version(package_name: str) -> str:
    """
    功能：读取可选依赖版本。

    Read optional package version with stable absent fallback.

    Args:
        package_name: Python package name.

    Returns:
        Version string or "<absent>".
    """
    if not isinstance(package_name, str) or not package_name:
        raise TypeError("package_name must be non-empty str")
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "<absent>"
