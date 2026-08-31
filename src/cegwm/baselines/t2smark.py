"""Method-faithful T2SMark codec and SD3.5 ordinary-RGB score interface.

This module reproduces the Tail-Truncated Sampling mechanism in the official
``src/t2s.py`` while keeping tensors on their caller-provided device.  It does
not load a model, make a threshold decision, or provide a synthetic score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Sequence

import numpy as np
import torch


DEFAULT_KEY_LENGTH = 16
DEFAULT_MESSAGE_LENGTH = 256
DEFAULT_TAU = 0.674
SD35_LATENT_SHAPE = (1, 16, 64, 64)
KEY_CHANNELS = (0, 1, 2, 3)
MESSAGE_CHANNELS = tuple(range(4, 16))
DEFAULT_NUM_INVERSION_STEPS = 10


def _normal_cdf_negative(tau: float) -> float:
    return 0.5 * math.erfc(tau / math.sqrt(2.0))


def _validate_bits(bits: torch.Tensor | Sequence[int], length: int | None, *, name: str, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(bits, device=device)
    if value.ndim != 1 or value.numel() == 0 or (length is not None and value.numel() != length):
        required = "a non-empty" if length is None else f"a {length}-bit"
        raise ValueError(f"{name} must be {required} one-dimensional binary vector")
    if value.dtype == torch.bool:
        value = value.to(torch.int64)
    elif not value.dtype.is_floating_point and value.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        raise TypeError(f"{name} must contain binary integer or boolean values")
    if not bool(torch.all((value == 0) | (value == 1))):
        raise ValueError(f"{name} must contain only 0 and 1")
    return value.to(torch.int64)


def _bits_to_int(bits: torch.Tensor) -> int:
    return int(reduce(lambda accumulator, bit: accumulator * 2 + int(bit), bits.tolist(), 0))


@dataclass(frozen=True)
class T2SMarkCodec:
    """Official T2SMark Tail-Truncated Sampling codec on one latent partition."""

    message_length: int
    tau: float
    latent_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.message_length <= 0 or len(self.latent_shape) != 3 or any(dimension <= 0 for dimension in self.latent_shape):
            raise ValueError("message length and latent shape must be positive")
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be a positive finite value")
        if self.repeat_count <= 0 or self.codeword_length > self.noise_size:
            raise ValueError("T2SMark parameters produce an invalid tail-truncated codeword")

    @property
    def noise_size(self) -> int:
        return int(reduce(mul, self.latent_shape, 1))

    @property
    def repeat_count(self) -> int:
        return int(2.0 * _normal_cdf_negative(self.tau) * self.noise_size / self.message_length)

    @property
    def codeword_length(self) -> int:
        return self.message_length * self.repeat_count

    def _support_and_signs(self, key_bits: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = _validate_bits(key_bits, None, name="key_bits", device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(_bits_to_int(key))
        signs = torch.randint(0, 2, (self.codeword_length,), generator=generator, device=device, dtype=torch.int64)
        support = torch.randperm(self.noise_size, generator=generator, device=device)[:self.codeword_length]
        return support, signs.to(dtype=dtype).mul(2).sub(1)

    def encode(self, bits: torch.Tensor | Sequence[int], key_bits: torch.Tensor | Sequence[int], *, base_noise: torch.Tensor | None = None) -> torch.Tensor:
        """Encode bits using official tail/central magnitude rearrangement.

        ``base_noise`` supplies the same clean SD3.5 latent partition for a
        clean/watermarked pair.  Without it, this matches the official fresh
        Gaussian-noise source behavior on the selected device.
        """

        if base_noise is None:
            tensor_input = bits if isinstance(bits, torch.Tensor) else key_bits if isinstance(key_bits, torch.Tensor) else None
            device = tensor_input.device if tensor_input is not None else torch.device("cpu")
            dtype = torch.get_default_dtype()
            source = torch.randn(self.latent_shape, device=device, dtype=dtype)
        else:
            if not isinstance(base_noise, torch.Tensor) or tuple(base_noise.shape) != self.latent_shape:
                raise ValueError(f"base_noise must have shape {self.latent_shape}")
            if not base_noise.dtype.is_floating_point:
                raise TypeError("base_noise must have a floating dtype")
            device, dtype, source = base_noise.device, base_noise.dtype, base_noise
        message = _validate_bits(bits, self.message_length, name="bits", device=device)
        support, keyed_signs = self._support_and_signs(key_bits, device=device, dtype=dtype)
        selector = torch.zeros(self.noise_size, dtype=torch.bool, device=device)
        selector[support] = True
        repeated_bits = (1 - 2 * message).repeat(self.repeat_count).to(dtype=dtype)
        codeword = repeated_bits * keyed_signs
        flattened = source.flatten()
        tail = torch.topk(flattened.abs(), k=self.codeword_length, largest=True, sorted=False)
        central = torch.topk(flattened.abs(), k=self.noise_size - self.codeword_length, largest=False, sorted=False)
        encoded = torch.empty_like(flattened)
        encoded[selector] = tail.values * codeword
        central_signs = torch.randint(0, 2, (self.noise_size - self.codeword_length,), device=device, dtype=torch.int64)
        encoded[~selector] = central.values * central_signs.to(dtype=dtype).mul(2).sub(1)
        return encoded.reshape(self.latent_shape)

    def decode(self, reversed_noise: torch.Tensor, key_bits: torch.Tensor | Sequence[int], *, detection: bool = False) -> torch.Tensor | tuple[torch.Tensor, float]:
        """Decode the official bit estimate and optional native L1 score."""

        if not isinstance(reversed_noise, torch.Tensor) or tuple(reversed_noise.shape) != self.latent_shape:
            raise ValueError(f"reversed_noise must have shape {self.latent_shape}")
        if not reversed_noise.dtype.is_floating_point:
            raise TypeError("reversed_noise must have a floating dtype")
        support, keyed_signs = self._support_and_signs(key_bits, device=reversed_noise.device, dtype=reversed_noise.dtype)
        selector = torch.zeros(self.noise_size, dtype=torch.bool, device=reversed_noise.device)
        selector[support] = True
        selected = reversed_noise.flatten()[selector] * keyed_signs
        p = selected.reshape(self.repeat_count, self.message_length).sum(dim=0)
        decoded = (p < 0).to(torch.int64)
        if detection:
            return decoded, float(torch.linalg.vector_norm(p.flatten(), ord=1).item())
        return decoded


def t2smark_sd35_codecs() -> tuple[T2SMarkCodec, T2SMarkCodec]:
    """Return the official SD3.5 key and message codecs."""

    return (
        T2SMarkCodec(DEFAULT_KEY_LENGTH, DEFAULT_TAU, (4, 64, 64)),
        T2SMarkCodec(DEFAULT_MESSAGE_LENGTH, DEFAULT_TAU, (12, 64, 64)),
    )


def embed_t2smark_sd35(
    base_latent: torch.Tensor,
    master_key_bits: torch.Tensor | Sequence[int],
    session_key_bits: torch.Tensor | Sequence[int],
    message_bits: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Embed official T2SMark key/message partitions into one SD3.5 base latent."""

    if not isinstance(base_latent, torch.Tensor) or tuple(base_latent.shape) != SD35_LATENT_SHAPE:
        raise ValueError(f"base_latent must have shape {SD35_LATENT_SHAPE}")
    if not base_latent.dtype.is_floating_point:
        raise TypeError("base_latent must have a floating dtype")
    key_codec, message_codec = t2smark_sd35_codecs()
    result = base_latent.clone()
    result[0, KEY_CHANNELS] = key_codec.encode(session_key_bits, master_key_bits, base_noise=base_latent[0, KEY_CHANNELS])
    result[0, MESSAGE_CHANNELS] = message_codec.encode(message_bits, session_key_bits, base_noise=base_latent[0, MESSAGE_CHANNELS])
    return result


def _require_rgb_uint8(rgb: Any) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TypeError("T2SMark RGB detector requires HxWx3 uint8 RGB")
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("T2SMark RGB detector requires non-empty RGB dimensions")
    return rgb


def score_t2smark_rgb(
    rgb: Any,
    pipeline: Any,
    master_key_bits: torch.Tensor | Sequence[int],
    num_inversion_steps: int = DEFAULT_NUM_INVERSION_STEPS,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> float:
    """Return the official image-only ``norm1_w`` continuous score.

    The detector accepts only ordinary RGB, an official-compatible SD3.5
    inversion pipeline, and the master detection key. It makes no threshold
    decision and has no embed-side, prompt, or original-image input.
    """

    source = _require_rgb_uint8(rgb)
    if pipeline is None or not callable(getattr(pipeline, "get_image_latents", None)) or not callable(getattr(pipeline, "naive_forward_diffusion", None)):
        raise TypeError("pipeline must provide get_image_latents and naive_forward_diffusion")
    if not isinstance(num_inversion_steps, int) or isinstance(num_inversion_steps, bool) or num_inversion_steps <= 0:
        raise ValueError("num_inversion_steps must be a positive integer")
    resolved_device = torch.device(device) if device is not None else torch.device(getattr(pipeline, "_execution_device", "cpu"))
    resolved_dtype = dtype if dtype is not None else (torch.float16 if resolved_device.type == "cuda" else torch.float32)
    image = torch.from_numpy(source.copy()).permute(2, 0, 1).unsqueeze(0).to(device=resolved_device, dtype=resolved_dtype)
    image = image.div(255.0).mul(2.0).sub(1.0)
    latents = pipeline.get_image_latents(image, sample=False)
    reversed_latents = pipeline.naive_forward_diffusion(latents=latents, num_inference_steps=num_inversion_steps)
    if not isinstance(reversed_latents, torch.Tensor) or tuple(reversed_latents.shape) != SD35_LATENT_SHAPE:
        raise ValueError(f"pipeline must return reversed latents with shape {SD35_LATENT_SHAPE}")
    key_codec, _ = t2smark_sd35_codecs()
    _, score = key_codec.decode(reversed_latents[0, KEY_CHANNELS], master_key_bits, detection=True)
    if not math.isfinite(score):
        raise RuntimeError("T2SMark native continuous score must be finite")
    return score
