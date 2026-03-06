"""
测试用例：detect 侧 LF 分支选择与 ecc 双语义兼容。

功能说明：
- 验证 ecc="sparse_ldpc" 时 detect 侧优先使用 LFCoderPRC latent 路径。
- 验证 ecc=int 时 detect 侧使用 image DCT fallback 路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
from PIL import Image

from main.watermarking.detect import orchestrator as detect_orchestrator


def test_detect_lf_prefers_prc_path_for_sparse_ldpc() -> None:
    """
    功能：验证 sparse_ldpc 走 latent PRC 检测分支。

    Verify that detect LF path uses latent PRC branch when ecc="sparse_ldpc".

    Args:
        None.

    Returns:
        None.
    """
    cfg: Dict[str, Any] = {
        "watermark": {
            "key_id": "k1",
            "pattern_id": "p1",
            "lf": {
                "enabled": True,
                "ecc": "sparse_ldpc",
                "message_length": 8,
                "bp_iterations": 6,
                "variance": 1.5,
            },
            "hf": {"enabled": False},
        },
        "__detect_final_latents__": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
    }

    lf_score, hf_score, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record=None,
        plan_payload={"plan": {"band_spec": {}}},
        plan_digest="plan_digest_for_test",
        cfg_digest="cfg_digest_for_test",
    )

    assert isinstance(traces, dict)
    traces_dict: Dict[str, Any] = traces
    lf_node = traces_dict.get("lf")
    lf_trace: Dict[str, Any] = cast(Dict[str, Any], lf_node) if isinstance(lf_node, dict) else {}
    assert lf_trace.get("lf_detect_path") == "lf_coder_prc_latent"
    assert lf_trace.get("lf_status") in {"ok", "failed", "absent"}
    assert hf_score is None

    if lf_trace.get("lf_status") == "ok":
        assert isinstance(lf_score, float)


def test_detect_lf_uses_image_dct_for_int_ecc(tmp_path: Path) -> None:
    """
    功能：验证 int ecc 走 image DCT fallback 检测分支。

    Verify that detect LF path uses image DCT fallback branch when ecc is int.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.
    """
    image_array = np.zeros((16, 16, 3), dtype=np.uint8)
    image_path = tmp_path / "detect_input.png"
    Image.fromarray(image_array).save(image_path)

    cfg: Dict[str, Any] = {
        "watermark": {
            "key_id": "k2",
            "pattern_id": "p2",
            "lf": {
                "enabled": True,
                "ecc": 3,
                "strength": 0.5,
                "variance": 1.5,
            },
            "hf": {"enabled": False},
        }
    }

    lf_score, hf_score, traces = detect_orchestrator._extract_content_raw_scores_from_image(  # pyright: ignore[reportPrivateUsage]
        cfg=cfg,
        input_record={"image_path": str(image_path)},
        plan_payload={"plan": {"band_spec": {}}},
        plan_digest="plan_digest_for_test",
        cfg_digest="cfg_digest_for_test",
    )

    assert isinstance(traces, dict)
    traces_dict: Dict[str, Any] = traces
    lf_node = traces_dict.get("lf")
    lf_trace: Dict[str, Any] = cast(Dict[str, Any], lf_node) if isinstance(lf_node, dict) else {}
    assert lf_trace.get("lf_detect_path") == "image_dct_fallback"
    assert lf_trace.get("lf_status") in {"ok", "absent", "failed"}
    assert hf_score is None

    if lf_trace.get("lf_status") == "ok":
        assert isinstance(lf_score, float)


def test_bind_raw_scores_keeps_lf_status_and_appends_prc_latent_status() -> None:
    """
    功能：验证 raw score 绑定不会覆写 lf_status，并追加 prc_latent_status。 

    Verify raw-score binding keeps lf_status semantics and appends prc_latent_status.

    Args:
        None.

    Returns:
        None.
    """
    payload: Dict[str, Any] = {
        "status": "ok",
        "score_parts": {
            "lf_status": "ok",
        },
    }
    traces: Dict[str, Any] = {
        "lf": {
            "lf_status": "failed",
            "lf_failure_reason": "lf_insufficient_latent_dimension",
        },
        "hf": {
            "hf_status": "absent",
            "hf_absent_reason": "hf_disabled_by_config",
        },
    }

    detect_orchestrator._bind_raw_scores_to_content_payload(  # pyright: ignore[reportPrivateUsage]
        content_evidence_payload=payload,
        lf_score=None,
        hf_score=None,
        traces=traces,
    )

    score_parts = payload.get("score_parts")
    assert isinstance(score_parts, dict)
    score_parts_dict: Dict[str, Any] = score_parts
    assert score_parts_dict.get("lf_status") == "ok"
    assert score_parts_dict.get("prc_latent_status") == "failed"


def test_detect_mask_digest_passthrough_from_input_record_when_missing() -> None:
    """
    功能：验证 detect 成功且缺失 mask_digest 时会从 input_record 透传。 

    Verify detect mask_digest is passed through from input_record when content status is ok.

    Args:
        None.

    Returns:
        None.
    """
    payload: Dict[str, Any] = {
        "status": "ok",
        "score": 0.77,
        "mask_digest": None,
    }
    input_record: Dict[str, Any] = {
        "content_evidence": {
            "mask_digest": "abc123maskdigest",
        }
    }

    detect_orchestrator._populate_detect_mask_digest_from_input_record(  # pyright: ignore[reportPrivateUsage]
        content_evidence_payload=payload,
        input_record=input_record,
    )
    assert payload.get("mask_digest") == "abc123maskdigest"
