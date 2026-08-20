"""Frozen public RGB8 attacks for Stage-A contrastive LF attribution."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import ceil, cos, log, pi, sin, sqrt
from typing import Sequence

import numpy as np
import torch
from torch.nn.functional import conv2d as apply_channelwise_convolution
from PIL import Image, __version__ as pillow_version

ATTACK_IDS = (
    "identity",
    "jpeg_quality_70",
    "gaussian_blur_sigma_1",
    "gaussian_noise_sigma_0_01",
)


class ContrastiveLfAttackError(ValueError):
    pass


def _stable_json_utf8(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ContrastiveLfAttackResult:
    attack_id: str
    image_rgb8: torch.Tensor
    source_rgb8_digest: str
    attacked_rgb8_digest: str
    attack_identity: str


def _validate_rgb8(image: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(image, torch.Tensor)
        or image.dtype is not torch.uint8
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or not image.is_contiguous()
    ):
        raise ContrastiveLfAttackError("attack input must be contiguous [1,3,H,W] RGB8")
    return image.detach().to(device="cpu").contiguous()


def _digest(image: torch.Tensor) -> str:
    return sha256(bytes(image.reshape(-1).tolist())).hexdigest()


def _jpeg(image: torch.Tensor, quality: int) -> tuple[torch.Tensor, str]:
    if pillow_version != "12.3.0" or quality not in {70, 90}:
        raise ContrastiveLfAttackError("JPEG execution capability is unavailable")
    array = image[0].permute(1, 2, 0).contiguous().numpy()
    if array.dtype != np.uint8 or not array.flags.c_contiguous:
        raise ContrastiveLfAttackError("JPEG input layout drifted")
    handle = BytesIO()
    Image.fromarray(array, mode="RGB").save(
        handle,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    encoded = handle.getvalue()
    try:
        decoded = np.array(
            Image.open(BytesIO(encoded)).convert("RGB"),
            dtype=np.uint8,
            copy=True,
            order="C",
        )
    except Exception as exc:
        raise ContrastiveLfAttackError("JPEG decode failed") from exc
    decoded = np.ascontiguousarray(decoded)
    if decoded.shape != array.shape or decoded.dtype != np.uint8:
        raise ContrastiveLfAttackError("JPEG output identity drifted")
    return torch.from_numpy(decoded).permute(2, 0, 1).unsqueeze(0).contiguous(), sha256(encoded).hexdigest()


def validate_jpeg_capability() -> dict[str, str]:
    fixture = np.empty((5, 7, 3), dtype=np.uint8)
    for row in range(5):
        for column in range(7):
            fixture[row, column] = (
                (37 * row + 19 * column + 11) % 256,
                (13 * row + 53 * column + 29) % 256,
                (71 * row + 7 * column + 47) % 256,
            )
    tensor = torch.from_numpy(fixture).permute(2, 0, 1).unsqueeze(0).contiguous()
    candidate_attack_rgb8, candidate_attack_encoded_sha256 = _jpeg(tensor, 70)
    stability_probe_rgb8, stability_probe_encoded_sha256 = _jpeg(tensor, 90)
    observed = {
        "fixture": sha256(fixture.tobytes(order="C")).hexdigest(),
        "jpeg_candidate_attack_encoded": candidate_attack_encoded_sha256,
        "jpeg_candidate_attack_decoded": sha256(
            candidate_attack_rgb8[0].permute(1, 2, 0).contiguous().numpy().tobytes(order="C")
        ).hexdigest(),
        "jpeg_stability_probe_encoded": stability_probe_encoded_sha256,
        "jpeg_stability_probe_decoded": sha256(
            stability_probe_rgb8[0].permute(1, 2, 0).contiguous().numpy().tobytes(order="C")
        ).hexdigest(),
    }
    expected = {
        "fixture": "d0d6b5c216be3f18108fc5033550da26fa2737a53852e8e2bbc04f3795b76194",
        "jpeg_candidate_attack_encoded": "9a202effd37e2b693f70fad3e9e01bc41d68df1fab4138de349a39963b49c80b",
        "jpeg_candidate_attack_decoded": "e4c5fc8268dce4b00f6b36bdcf542968bdba495a153cc273e9ece7279dd7029b",
        "jpeg_stability_probe_encoded": "e8c89254351499471fe086704f3ecc6e2fb76f7cccd33e5a8ecc2fc33fc54c36",
        "jpeg_stability_probe_decoded": "0d0e02529511b7cae4de3479b9645569c3985750655217abfb6c7d6326b362c7",
    }
    if observed != expected:
        raise ContrastiveLfAttackError("JPEG golden capability mismatch")
    return observed


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(ceil(3.0 * sigma))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    gaussian_kernel_binary64 = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    gaussian_kernel_binary64 /= np.sum(gaussian_kernel_binary64, dtype=np.float64)
    kernel = torch.tensor(gaussian_kernel_binary64.astype(np.float32), dtype=torch.float32)
    value = image.to(dtype=torch.float32)
    horizontal = torch.nn.functional.pad(value, (radius, radius, 0, 0), mode="reflect")
    horizontal = apply_channelwise_convolution(
        horizontal,
        kernel.reshape(1, 1, 1, -1).repeat(3, 1, 1, 1),
        groups=3,
    )
    vertical = torch.nn.functional.pad(horizontal, (0, 0, radius, radius), mode="reflect")
    vertical = apply_channelwise_convolution(
        vertical,
        kernel.reshape(1, 1, -1, 1).repeat(3, 1, 1, 1),
        groups=3,
    )
    return torch.round(torch.clamp(vertical, 0.0, 255.0)).to(torch.uint8).contiguous()


def _noise_values(count: int, identity: dict[str, object]) -> torch.Tensor:
    seed = sha256(_stable_json_utf8(identity)).digest()
    values: list[float] = []
    counter = 0
    denominator = float(1 << 64)
    while len(values) < count:
        block = sha256(seed + counter.to_bytes(8, "big")).digest()
        for offset in range(0, len(block), 16):
            first_uniform = (int.from_bytes(block[offset : offset + 8], "big") + 0.5) / denominator
            second_uniform = (int.from_bytes(block[offset + 8 : offset + 16], "big") + 0.5) / denominator
            radius = sqrt(-2.0 * log(first_uniform))
            angle = 2.0 * pi * second_uniform
            values.extend((radius * cos(angle), radius * sin(angle)))
            if len(values) >= count:
                break
        counter += 1
    return torch.tensor(values[:count], dtype=torch.float32)


def apply_contrastive_lf_attack(
    image_rgb8: torch.Tensor,
    attack_id: str,
    *,
    source_cluster_id: str,
    generation_seed: int,
) -> ContrastiveLfAttackResult:
    image = _validate_rgb8(image_rgb8)
    if attack_id not in ATTACK_IDS or type(source_cluster_id) is not str or not source_cluster_id or type(generation_seed) is not int:
        raise ContrastiveLfAttackError("attack identity is invalid")
    source_digest = _digest(image)
    extra: dict[str, object] = {}
    if attack_id == "identity":
        attacked = image.clone()
    elif attack_id == "jpeg_quality_70":
        attacked, extra_digest = _jpeg(image, 70)
        extra["encoded_sha256"] = extra_digest
    elif attack_id == "gaussian_blur_sigma_1":
        attacked = _blur(image, 1.0)
    else:
        noise_identity = {
            "attack_id": attack_id,
            "generation_seed": generation_seed,
            "protocol_id": "contrastive_lf_branch_attribution",
            "source_cluster_id": source_cluster_id,
        }
        noise = _noise_values(image.numel(), noise_identity).reshape(image.shape)
        attacked = torch.round(
            torch.clamp(image.to(torch.float32) + noise * (0.01 * 255.0), 0.0, 255.0)
        ).to(torch.uint8).contiguous()
        extra["public_attack_noise_digest_random"] = sha256(
            noise.numpy().tobytes(order="C")
        ).hexdigest()
    attacked_digest = _digest(attacked)
    identity = sha256(
        _stable_json_utf8(
            {
                "attack_id": attack_id,
                "attacked_rgb8_digest": attacked_digest,
                "generation_seed": generation_seed,
                "source_cluster_id": source_cluster_id,
                "source_rgb8_digest": source_digest,
                **extra,
            }
        )
    ).hexdigest()
    return ContrastiveLfAttackResult(
        attack_id=attack_id,
        image_rgb8=attacked,
        source_rgb8_digest=source_digest,
        attacked_rgb8_digest=attacked_digest,
        attack_identity=identity,
    )


__all__ = [
    "ATTACK_IDS",
    "ContrastiveLfAttackError",
    "ContrastiveLfAttackResult",
    "apply_contrastive_lf_attack",
    "validate_jpeg_capability",
]
