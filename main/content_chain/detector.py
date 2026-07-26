"""当前冻结的 HF-only CEG-WM content detector。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from main.shared.key_schedule import stable_json_utf8

from .hf_detector import HfDetectionResult


class ContentDetectorError(ValueError):
    """HF-only 内容分支结果或身份无效。"""


@dataclass(frozen=True, slots=True)
class HfOnlyContentDetectionResult:
    """保留完整 HF 分支结果的当前 `D_M` 输出。"""

    candidate_id: str
    content_score: float
    detector_identity: str
    content_config_digest: str
    hf_result: HfDetectionResult


def content_detector(
    hf_result: HfDetectionResult,
) -> HfOnlyContentDetectionResult:
    """只消费独立 `s_hf`，形成当前 HF-only `D_M`。"""

    if type(hf_result) is not HfDetectionResult:
        raise ContentDetectorError(
            "HF-only content detector requires HfDetectionResult"
        )
    if hf_result.candidate_id != "hf_sparse_tail":
        raise ContentDetectorError("HF candidate identity mismatch")
    if not isfinite(hf_result.hf_score):
        raise ContentDetectorError("HF score must be finite")

    content_config = {
        "branch_detector_identity": hf_result.detector_identity,
        "candidate_id": "hf_sparse_tail",
        "content_detector_role": "hf_only_direct_score",
        "lf_combination_enabled": False,
    }
    content_config_digest = sha256(
        stable_json_utf8(content_config)
    ).hexdigest()
    detector_identity_value = {
        "branch_detector_identity": hf_result.detector_identity,
        "content_config_digest": content_config_digest,
        "detector_name": "ceg_wm_hf_only_content_detector",
    }
    return HfOnlyContentDetectionResult(
        candidate_id="hf_sparse_tail",
        content_score=hf_result.hf_score,
        detector_identity=sha256(
            stable_json_utf8(detector_identity_value)
        ).hexdigest(),
        content_config_digest=content_config_digest,
        hf_result=hf_result,
    )
