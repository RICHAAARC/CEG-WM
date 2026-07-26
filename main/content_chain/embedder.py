"""CEG-WM 内容写入责任；批次 2 只开放 HF-only 模式。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from struct import pack, unpack
from typing import Sequence

from main.shared.key_schedule import stable_json_utf8

from .hf_carrier import HfCarrierResult

CONTENT_RELATIVE_L2_NUMERATOR = 3
CONTENT_RELATIVE_L2_DENOMINATOR = 250


class ContentEmbedderError(ValueError):
    """HF-only 内容方向或共同总预算无效。"""


@dataclass(frozen=True, slots=True)
class HfOnlyEmbeddingResult:
    """未物化 dtype 的 HF-only 理论内容更新。"""

    candidate_id: str
    mode: str
    shape: tuple[int, int, int, int]
    delta_content: tuple[float, ...]
    delta_content_digest: str
    latent_norm: float
    target_total_norm: float
    target_relative_l2: float
    carrier_config_digest: str
    embedder_config_digest: str


def _float32(value: float) -> float:
    if not isfinite(value):
        raise ContentEmbedderError("content embedder value must be finite")
    try:
        rounded = unpack(">f", pack(">f", value))[0]
    except (OverflowError, ValueError) as exc:
        raise ContentEmbedderError(
            "content embedder value is outside binary32 range"
        ) from exc
    if not isfinite(rounded):
        raise ContentEmbedderError("content embedder binary32 value must be finite")
    return rounded


def _vector(values: Sequence[float], size: int, role: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContentEmbedderError(f"{role} must be a numeric sequence")
    if len(values) != size:
        raise ContentEmbedderError(f"{role} length does not match carrier shape")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContentEmbedderError(f"{role} must contain only finite numbers")
        converted.append(_float32(float(value)))
    return tuple(converted)


def _l2_norm(values: Sequence[float]) -> float:
    squared_sum = 0.0
    for value in values:
        squared_sum = _float32(squared_sum + _float32(value * value))
    return _float32(sqrt(squared_sum))


def _digest(values: Sequence[float]) -> str:
    return sha256(b"".join(pack(">f", value) for value in values)).hexdigest()


def content_embedder(
    latent_values: Sequence[float],
    hf_carrier_result: HfCarrierResult,
) -> HfOnlyEmbeddingResult:
    """按共同 `rho_content=0.012` 预算产生 HF-only `delta_content`。"""

    if type(hf_carrier_result) is not HfCarrierResult:
        raise ContentEmbedderError("HF-only embedder requires HfCarrierResult")
    element_count = len(hf_carrier_result.direction)
    latent = _vector(latent_values, element_count, "latent_values")
    direction = _vector(
        hf_carrier_result.direction,
        element_count,
        "hf_direction",
    )
    if _digest(direction) != hf_carrier_result.direction_digest:
        raise ContentEmbedderError("HF carrier direction digest mismatch")

    latent_norm = _l2_norm(latent)
    if latent_norm == 0.0:
        raise ContentEmbedderError("callback latent has zero L2 energy")
    direction_norm = _l2_norm(direction)
    if direction_norm == 0.0:
        raise ContentEmbedderError("HF direction has zero L2 energy")
    if abs(direction_norm - 1.0) > 1e-5:
        raise ContentEmbedderError("HF carrier direction must be unit L2")

    target_relative_l2 = _float32(
        CONTENT_RELATIVE_L2_NUMERATOR / CONTENT_RELATIVE_L2_DENOMINATOR
    )
    target_total_norm = _float32(target_relative_l2 * latent_norm)
    if target_total_norm == 0.0:
        raise ContentEmbedderError("target content update vanished in float32")
    delta_content = tuple(
        _float32(target_total_norm * value) for value in direction
    )
    if _l2_norm(delta_content) == 0.0:
        raise ContentEmbedderError("HF-only delta_content has zero L2 energy")

    identity = {
        "candidate_ids": ["runtime_sd35_flowmatch", "hf_sparse_tail"],
        "carrier_config_digest": hf_carrier_result.carrier_config_digest,
        "mode": "hf_only",
        "rho_content_denominator": CONTENT_RELATIVE_L2_DENOMINATOR,
        "rho_content_numerator": CONTENT_RELATIVE_L2_NUMERATOR,
        "shape": list(hf_carrier_result.shape),
    }
    return HfOnlyEmbeddingResult(
        candidate_id="hf_sparse_tail",
        mode="hf_only",
        shape=hf_carrier_result.shape,
        delta_content=delta_content,
        delta_content_digest=_digest(delta_content),
        latent_norm=latent_norm,
        target_total_norm=target_total_norm,
        target_relative_l2=target_relative_l2,
        carrier_config_digest=hf_carrier_result.carrier_config_digest,
        embedder_config_digest=sha256(stable_json_utf8(identity)).hexdigest(),
    )
