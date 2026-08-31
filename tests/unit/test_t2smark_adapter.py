"""CPU fixtures for the T2SMark codec and its RGB scoring boundary."""

import inspect
import math

import numpy as np
import pytest
import torch

from cegwm.baselines import (
    DEFAULT_NUM_INVERSION_STEPS,
    KEY_CHANNELS,
    MESSAGE_CHANNELS,
    SD35_LATENT_SHAPE,
    T2SMarkCodec,
    embed_t2smark_sd35,
    score_t2smark_rgb,
    t2smark_sd35_codecs,
)
from cegwm.baselines.adapters import adapter_plan


def _reference_encode(bits: torch.Tensor, key: torch.Tensor, base: torch.Tensor, tau: float) -> torch.Tensor:
    """Independent transcription of the published T2SMark encode formula."""
    message_length = int(bits.numel())
    noise_size = int(base.numel())
    repeat_count = int(math.erfc(tau / math.sqrt(2.0)) * noise_size / message_length)
    generator = torch.Generator()
    seed = 0
    for bit in key.tolist():
        seed = 2 * seed + int(bit)
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (message_length * repeat_count,), generator=generator, device="cpu")
    signs = signs.to(device=base.device, dtype=base.dtype).mul(2).sub(1)
    support = torch.randperm(noise_size, generator=generator, device="cpu")[: message_length * repeat_count]
    support = support.to(device=base.device)
    selector = torch.zeros(noise_size, device=base.device, dtype=torch.bool)
    selector[support] = True
    codeword = (1 - 2 * bits).repeat(repeat_count).to(dtype=base.dtype) * signs
    flattened = base.flatten()
    tail = torch.topk(flattened.abs(), k=message_length * repeat_count, largest=True, sorted=False)
    central = torch.topk(flattened.abs(), k=noise_size - message_length * repeat_count, largest=False, sorted=False)
    encoded = flattened.clone()
    encoded[selector] = tail.values * codeword
    central_signs = torch.randint(0, 2, (noise_size - message_length * repeat_count,), device=base.device)
    encoded[~selector] = central.values * central_signs.to(dtype=base.dtype).mul(2).sub(1)
    return encoded.reshape_as(base)


def test_codec_matches_published_encode_formula_on_cpu_fixture() -> None:
    codec = T2SMarkCodec(message_length=4, tau=0.674, latent_shape=(2, 4, 4))
    bits = torch.tensor([0, 1, 0, 1])
    key = torch.tensor([1, 0, 1, 1])
    base = torch.linspace(-2.0, 2.0, steps=32).reshape(2, 4, 4)

    torch.manual_seed(31415)
    actual = codec.encode(bits, key, base_noise=base)
    torch.manual_seed(31415)
    expected = _reference_encode(bits, key, base, codec.tau)

    torch.testing.assert_close(actual, expected)


def test_codec_uses_official_cpu_keyed_prng_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = T2SMarkCodec(message_length=4, tau=0.674, latent_shape=(2, 4, 4))
    original_generator = torch.Generator
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_generator(*args: object, **kwargs: object) -> torch.Generator:
        calls.append((args, kwargs))
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", record_generator)
    codec.encode(torch.zeros(4, dtype=torch.int64), torch.tensor([1, 0, 1, 1]), base_noise=torch.ones(2, 4, 4))

    assert calls == [((), {})]


def test_codec_round_trip_and_detection_score_for_uniform_message() -> None:
    codec = T2SMarkCodec(message_length=4, tau=0.674, latent_shape=(2, 4, 4))
    bits = torch.zeros(4, dtype=torch.int64)
    key = torch.tensor([1, 0, 0, 1])
    base = torch.linspace(-3.0, 3.0, steps=32).reshape(2, 4, 4)

    torch.manual_seed(99)
    encoded = codec.encode(bits, key, base_noise=base)
    decoded, score = codec.decode(encoded, key, detection=True)

    assert torch.equal(decoded, bits)
    assert math.isfinite(score)
    assert score > 0.0


def test_sd35_embedding_preserves_channel_partition_and_codec_defaults() -> None:
    key_codec, message_codec = t2smark_sd35_codecs()
    assert key_codec.latent_shape == (4, 64, 64)
    assert message_codec.latent_shape == (12, 64, 64)
    assert KEY_CHANNELS == (0, 1, 2, 3)
    assert MESSAGE_CHANNELS == tuple(range(4, 16))

    base = torch.linspace(-1.0, 1.0, steps=int(np.prod(SD35_LATENT_SHAPE))).reshape(SD35_LATENT_SHAPE)
    master = torch.zeros(16, dtype=torch.int64)
    session = torch.zeros(16, dtype=torch.int64)
    message = torch.zeros(256, dtype=torch.int64)
    torch.manual_seed(7)
    embedded = embed_t2smark_sd35(base, master, session, message)

    assert embedded.shape == base.shape
    assert embedded.dtype == base.dtype
    assert embedded.data_ptr() != base.data_ptr()
    assert not torch.equal(embedded, base)
    decoded, score = key_codec.decode(embedded[0, KEY_CHANNELS], master, detection=True)
    assert torch.equal(decoded, session)
    assert score > 0.0


class _RecordingPipeline:
    _execution_device = "cpu"

    def __init__(self, reversed_latents: torch.Tensor) -> None:
        self.reversed_latents = reversed_latents
        self.image: torch.Tensor | None = None
        self.sample: bool | None = None
        self.steps: int | None = None

    def get_image_latents(self, image: torch.Tensor, sample: bool) -> torch.Tensor:
        self.image = image.detach().clone()
        self.sample = sample
        return torch.zeros(SD35_LATENT_SHAPE, dtype=image.dtype, device=image.device)

    def naive_forward_diffusion(self, *, latents: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        assert latents.shape == SD35_LATENT_SHAPE
        self.steps = num_inference_steps
        return self.reversed_latents.to(device=latents.device, dtype=latents.dtype)


def test_rgb_score_uses_only_ordinary_rgb_and_pipeline_inversion_boundary() -> None:
    key_codec, _ = t2smark_sd35_codecs()
    master = torch.zeros(16, dtype=torch.int64)
    reversed_latents = torch.zeros(SD35_LATENT_SHAPE)
    reversed_latents[0, KEY_CHANNELS] = key_codec.encode(
        torch.zeros(16, dtype=torch.int64), master, base_noise=torch.linspace(-1.0, 1.0, steps=4 * 64 * 64).reshape(4, 64, 64)
    )
    pipeline = _RecordingPipeline(reversed_latents)
    rgb = np.array([[[0, 127, 255], [255, 127, 0]]], dtype=np.uint8)

    score = score_t2smark_rgb(rgb, pipeline, master)

    assert isinstance(score, float)
    assert math.isfinite(score)
    assert pipeline.sample is False
    assert pipeline.steps == DEFAULT_NUM_INVERSION_STEPS
    assert pipeline.image is not None
    assert pipeline.image.shape == (1, 3, 1, 2)
    torch.testing.assert_close(pipeline.image[0, :, 0, 0], torch.tensor([-1.0, -1.0 / 255.0, 1.0]))


def test_rgb_score_rejects_non_ordinary_rgb_and_has_no_latent_escape_hatch() -> None:
    parameters = inspect.signature(score_t2smark_rgb).parameters
    assert not {"original", "prompt", "base_noise", "session_key_bits", "message_bits"}.intersection(parameters)
    pipeline = _RecordingPipeline(torch.zeros(SD35_LATENT_SHAPE))
    with pytest.raises(TypeError, match="uint8"):
        score_t2smark_rgb(np.zeros((2, 2, 3), dtype=np.float32), pipeline, torch.zeros(16, dtype=torch.int64))
    with pytest.raises(TypeError, match="HxWx3"):
        score_t2smark_rgb(np.zeros((2, 2), dtype=np.uint8), pipeline, torch.zeros(16, dtype=torch.int64))
    with pytest.raises(ValueError, match="16-bit"):
        score_t2smark_rgb(np.zeros((2, 2, 3), dtype=np.uint8), pipeline, torch.zeros(15, dtype=torch.int64))


def test_sd35_embedding_rejects_nonstandard_master_session_or_message_lengths() -> None:
    base = torch.zeros(SD35_LATENT_SHAPE)
    with pytest.raises(ValueError, match="16-bit"):
        embed_t2smark_sd35(base, torch.zeros(15, dtype=torch.int64), torch.zeros(16, dtype=torch.int64), torch.zeros(256, dtype=torch.int64))
    with pytest.raises(ValueError, match="16-bit"):
        embed_t2smark_sd35(base, torch.zeros(16, dtype=torch.int64), torch.zeros(17, dtype=torch.int64), torch.zeros(256, dtype=torch.int64))
    with pytest.raises(ValueError, match="256-bit"):
        embed_t2smark_sd35(base, torch.zeros(16, dtype=torch.int64), torch.zeros(16, dtype=torch.int64), torch.zeros(255, dtype=torch.int64))


def test_t2smark_adapter_plan_remains_gpu_unverified() -> None:
    plan = adapter_plan("t2smark")
    assert plan.execution_status == "adapter_interface_available_gpu_unverified"
    assert plan.blocker == "local GPU is unavailable; a Colab canary is required"
