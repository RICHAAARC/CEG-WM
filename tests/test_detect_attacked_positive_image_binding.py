"""
File purpose: Validate attacked-positive LF image-conditioned binding contracts.
Module type: General module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
from PIL import Image
import pytest

from main.diffusion.sd3.trajectory_tap import LatentTrajectoryCache
from main.watermarking.detect import orchestrator as detect_orchestrator


def _build_detect_cfg() -> Dict[str, Any]:
    """
    功能：构造最小 detect 配置夹具。 

    Build the minimal detect configuration fixture.

    Args:
        None.

    Returns:
        Detect configuration mapping.
    """
    return {
        "watermark": {
            "key_id": "k-test",
            "pattern_id": "p-test",
            "lf": {
                "enabled": True,
                "ecc": "sparse_ldpc",
            },
            "hf": {
                "enabled": False,
            },
        }
    }


def _build_plan_payload() -> Dict[str, Any]:
    """
    功能：构造最小 LF 计划夹具。 

    Build the minimal LF plan payload fixture.

    Args:
        None.

    Returns:
        Planner payload mapping.
    """
    return {
        "plan": {
            "lf_basis": {
                "trajectory_feature_spec": {
                    "feature_operator": "masked_normalized_random_projection",
                    "edit_timestep": 0,
                },
            },
            "band_spec": {},
        }
    }


def _build_latent_cache(fill_value: float) -> LatentTrajectoryCache:
    """
    功能：构造包含单个 timestep 的 latent cache。 

    Build a latent cache containing one deterministic timestep.

    Args:
        fill_value: Fill value used in the cached tensor.

    Returns:
        Latent trajectory cache fixture.
    """
    cache = LatentTrajectoryCache()
    cache.capture(0, np.full((1, 1, 2, 4), fill_value, dtype=np.float32))
    return cache


def _write_test_image(tmp_path: Path, file_name: str) -> Path:
    """
    功能：写入最小 detect 输入图像。 

    Write a minimal detect input image fixture.

    Args:
        tmp_path: Pytest temporary directory.
        file_name: Output file name.

    Returns:
        Written image path.
    """
    image_path = tmp_path / file_name
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(image_path)
    return image_path


def test_attacked_positive_lf_uses_image_conditioned_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 attacked_positive 的 LF content 分数绑定到 image-conditioned cache。 

    Ensure attacked-positive LF content scoring uses the image-conditioned cache
    rather than the default detect trajectory cache.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    attacked_image_path = _write_test_image(tmp_path, "attacked.png")
    clean_cache = _build_latent_cache(1.0)
    attacked_cache = _build_latent_cache(2.0)
    captures: List[Dict[str, Any]] = []

    def fake_extract_lf_raw_score_from_trajectory(**kwargs: Any) -> tuple[float, Dict[str, Any]]:
        captures.append(dict(kwargs))
        return 0.42, {
            "lf_status": "ok",
            "lf_detect_path": str(kwargs.get("detect_path")),
        }

    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        fake_extract_lf_raw_score_from_trajectory,
    )

    cfg = _build_detect_cfg()
    cfg["__detect_trajectory_latent_cache__"] = clean_cache
    cfg["__lf_attacked_image_conditioned_latent_cache__"] = attacked_cache
    cfg["__lf_formal_exact_context__"] = {
        "formal_exact_evidence_source": "input_image_conditioned_reconstruction",
        "formal_exact_object_binding_status": "ok",
        "formal_exact_image_path_source": "input_record.watermarked_path",
        "image_conditioned_reconstruction_available": True,
        "image_conditioned_reconstruction_status": "ok",
    }

    lf_score, lf_trace = detect_orchestrator._extract_attacked_positive_lf_raw_score(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record={
            "sample_role": "attacked_positive",
            "watermarked_path": attacked_image_path.as_posix(),
            "image_path": (tmp_path / "clean.png").as_posix(),
            "inputs": {
                "input_image_path": (tmp_path / "preview.png").as_posix(),
            },
        },
        plan_payload=_build_plan_payload(),
        plan_digest="plan-digest",
        cfg_digest="cfg-digest",
    )

    assert lf_score == 0.42
    assert len(captures) == 1
    assert captures[0]["latent_cache"] is attacked_cache
    assert captures[0]["detect_path"] == "low_freq_template_image_conditioned_attack"
    assert lf_trace["lf_detect_path"] == "low_freq_template_image_conditioned_attack"
    assert lf_trace["formal_exact_object_binding_status"] == "ok"
    assert lf_trace["image_conditioned_reconstruction_status"] == "ok"


def test_attacked_positive_lf_fail_closes_without_image_conditioned_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证 attacked_positive 缺失 image-conditioned cache 时不会回退到 clean trajectory。 

    Ensure attacked-positive LF scoring fails closed instead of falling back to
    the clean detect trajectory cache when the image-conditioned cache is unavailable.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    attacked_image_path = _write_test_image(tmp_path, "attacked_missing_cache.png")

    def fail_if_called(**kwargs: Any) -> tuple[float, Dict[str, Any]]:
        raise AssertionError("attacked_positive must not fall back to detect trajectory cache")

    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        fail_if_called,
    )

    cfg = _build_detect_cfg()
    cfg["__detect_trajectory_latent_cache__"] = _build_latent_cache(1.0)
    cfg["__lf_formal_exact_context__"] = {
        "formal_exact_object_binding_status": "absent",
        "image_conditioned_reconstruction_available": False,
        "image_conditioned_reconstruction_status": "detect_input_image_absent",
    }

    lf_score, lf_trace = detect_orchestrator._extract_attacked_positive_lf_raw_score(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record={
            "sample_role": "attacked_positive",
            "watermarked_path": attacked_image_path.as_posix(),
        },
        plan_payload=_build_plan_payload(),
        plan_digest="plan-digest",
        cfg_digest="cfg-digest",
    )

    assert lf_score is None
    assert lf_trace["lf_status"] == "absent"
    assert lf_trace["lf_absent_reason"] == "attack_image_conditioned_evidence_unavailable"
    assert lf_trace["lf_detect_path"] == "low_freq_template_image_conditioned_attack"
    assert lf_trace["image_conditioned_reconstruction_status"] == "detect_input_image_absent"


def test_clean_input_lf_keeps_default_trajectory_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    功能：验证非 attacked_positive 输入保持原有 trajectory LF 路径。 

    Ensure non-attacked-positive inputs keep the original trajectory LF path.

    Args:
        tmp_path: Pytest temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    clean_image_path = _write_test_image(tmp_path, "clean.png")
    captures: List[Dict[str, Any]] = []

    def fake_extract_lf_raw_score_from_trajectory(**kwargs: Any) -> tuple[float, Dict[str, Any]]:
        captures.append(dict(kwargs))
        return 0.11, {
            "lf_status": "ok",
            "lf_detect_path": str(kwargs.get("detect_path", "low_freq_template_trajectory")),
        }

    monkeypatch.setattr(
        detect_orchestrator,
        "_extract_lf_raw_score_from_trajectory",
        fake_extract_lf_raw_score_from_trajectory,
    )

    cfg = _build_detect_cfg()
    cfg["__detect_trajectory_latent_cache__"] = _build_latent_cache(1.0)
    cfg["__lf_attacked_image_conditioned_latent_cache__"] = _build_latent_cache(2.0)

    lf_score, hf_score, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record={
            "sample_role": "positive",
            "watermarked_path": clean_image_path.as_posix(),
        },
        plan_payload=_build_plan_payload(),
        plan_digest="plan-digest",
        cfg_digest="cfg-digest",
    )

    assert hf_score is None
    assert lf_score == 0.11
    assert len(captures) == 1
    assert captures[0].get("latent_cache") is None
    assert captures[0].get("detect_path") is None
    lf_trace = cast(Dict[str, Any], traces["lf"])
    assert lf_trace["lf_detect_path"] == "low_freq_template_trajectory"