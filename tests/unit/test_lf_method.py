from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method import frequency
from cegwm.method.frequency import radial_frequency_mask
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BALANCED_BLOCKS_RADIAL_BLOCKS,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    LF_BLOCKNORM_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_RADIAL_BLOCKS,
    LF_CORE_CANDIDATE_ID,
    LF_SHELL_CANDIDATE_ID,
    FrozenLFPublicAssets,
    _spec,
    inject_lf_carrier,
    reconstruct_lf_carrier,
    score_lf_image,
)


class _Processor:
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


class _Distribution:
    def __init__(self, value: torch.Tensor) -> None:
        self._value = value

    def mode(self) -> torch.Tensor:
        return self._value


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
        self.observation = torch.ones((1, 4, 32, 32), dtype=torch.float32)
        self.encode_calls = 0

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        del pixels
        self.encode_calls += 1
        return SimpleNamespace(latent_dist=_Distribution(self.observation))


def _assets(candidate_id: str, vae: _VAE | None = None) -> FrozenLFPublicAssets:
    return FrozenLFPublicAssets(
        vae=vae or _VAE(),
        image_processor=_Processor(),
        image_processor_id="sd35-vae-image-processor-v1",
        candidate_id=candidate_id,
    )


def _blocknorm_assets(vae: _VAE | None = None) -> FrozenLFPublicAssets:
    return FrozenLFPublicAssets(
        vae=vae or _VAE(),
        image_processor=_Processor(),
        image_processor_id="sd35-vae-image-processor-v1",
        candidate_id=LF_SHELL_CANDIDATE_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BLOCKNORM_EVALUATED_CANDIDATE_ID,
    )


def _balanced_assets(vae: _VAE | None = None) -> FrozenLFPublicAssets:
    return FrozenLFPublicAssets(
        vae=vae or _VAE(),
        image_processor=_Processor(),
        image_processor_id="sd35-vae-image-processor-v1",
        candidate_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        evaluated_candidate_id=LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )


@pytest.mark.unit
def test_core_and_shell_are_disjoint_exact_lf_partition() -> None:
    core_spec = _spec(_assets(LF_CORE_CANDIDATE_ID))
    shell_spec = _spec(_assets(LF_SHELL_CANDIDATE_ID))
    core = radial_frequency_mask(128, 128, core_spec)
    shell = radial_frequency_mask(128, 128, shell_spec)
    vertical = np.fft.fftfreq(128)[:, None]
    horizontal = np.fft.rfftfreq(128)[None, :]
    radius = np.hypot(vertical, horizontal) / np.hypot(0.5, 0.5)
    frozen_lf = (radius >= 0.04) & (radius <= 0.24)

    assert not np.any(core & shell)
    assert np.array_equal(core | shell, frozen_lf)
    assert core_spec.max_radius == shell_spec.min_radius == 0.14
    assert core_spec.max_inclusive is False
    assert shell_spec.max_inclusive is True
    assert not np.any(core[radius == 0.14])
    assert np.all(shell[radius == 0.14])


@pytest.mark.unit
def test_lf_carriers_bind_key_and_candidate_without_private_unit_identity() -> None:
    key = b"0123456789abcdef0123456789abcdef"
    core = reconstruct_lf_carrier(
        key, (1, 4, 32, 32), _assets(LF_CORE_CANDIDATE_ID), dtype=torch.float32, device="cpu"
    )
    core_repeat = reconstruct_lf_carrier(
        key, (1, 4, 32, 32), _assets(LF_CORE_CANDIDATE_ID), dtype=torch.float32, device="cpu"
    )
    shell = reconstruct_lf_carrier(
        key, (1, 4, 32, 32), _assets(LF_SHELL_CANDIDATE_ID), dtype=torch.float32, device="cpu"
    )
    wrong = reconstruct_lf_carrier(
        b"abcdef0123456789abcdef0123456789",
        (1, 4, 32, 32),
        _assets(LF_CORE_CANDIDATE_ID),
        dtype=torch.float32,
        device="cpu",
    )

    assert torch.equal(core, core_repeat)
    assert not torch.equal(core, shell)
    assert not torch.equal(core, wrong)
    assert torch.linalg.vector_norm(core.float()).item() == pytest.approx(1.0)


@pytest.mark.unit
def test_lf_prg_domain_contains_only_method_and_public_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domains: list[str] = []

    def fake_prg(
        key: object,
        domain: str,
        shape: tuple[int, ...],
        *,
        dtype: object,
    ) -> np.ndarray:
        del key, dtype
        domains.append(domain)
        return np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(frequency, "prg_rademacher", fake_prg)
    reconstruct_lf_carrier(
        b"0123456789abcdef0123456789abcdef",
        (1, 2, 32, 32),
        _assets(LF_CORE_CANDIDATE_ID),
        dtype=torch.float32,
        device="cpu",
    )

    assert domains == [
        "lf/lf_core_rademacher_v1/spatial-irfft2-real-rademacher-v1/channels=2/height=32/width=32/channel=0",
        "lf/lf_core_rademacher_v1/spatial-irfft2-real-rademacher-v1/channels=2/height=32/width=32/channel=1",
    ]
    assert not any(token in "/".join(domains) for token in ("unit", "prompt", "seed", "winner", "record"))

    domains.clear()
    reconstruct_lf_carrier(
        b"0123456789abcdef0123456789abcdef",
        (1, 2, 32, 32),
        _blocknorm_assets(),
        dtype=torch.float32,
        device="cpu",
    )
    assert domains == [
        "lf/lf_shell_rademacher_v1/spatial-irfft2-real-rademacher-v1/channels=2/height=32/width=32/channel=0",
        "lf/lf_shell_rademacher_v1/spatial-irfft2-real-rademacher-v1/channels=2/height=32/width=32/channel=1",
    ]
    assert LF_BLOCKNORM_EVALUATED_CANDIDATE_ID not in "/".join(domains)
    assert LF_BLOCKNORM_DETECTOR_STATISTIC_ID not in "/".join(domains)


@pytest.mark.unit
@pytest.mark.parametrize("candidate_id", [LF_CORE_CANDIDATE_ID, LF_SHELL_CANDIDATE_ID])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_each_lf_candidate_uses_full_single_actual_dtype_budget(
    candidate_id: str,
    dtype: torch.dtype,
) -> None:
    latents = torch.linspace(-1.0, 1.0, 4 * 32 * 32, dtype=dtype).reshape(1, 4, 32, 32)
    injected, measurement = inject_lf_carrier(
        latents,
        b"0123456789abcdef0123456789abcdef",
        _assets(candidate_id),
    )
    actual = torch.linalg.vector_norm(injected.double() - latents.double())
    base = torch.linalg.vector_norm(latents.double())

    assert injected.dtype is dtype
    assert not torch.equal(injected, latents)
    assert 0.0 < measurement.relative_l2 <= 0.012
    assert measurement.relative_l2 == pytest.approx((actual / base).item())


@pytest.mark.unit
def test_lf_blind_signature_and_final_image_reencode_are_keyed() -> None:
    assert tuple(inspect.signature(score_lf_image).parameters) == (
        "image",
        "detection_key",
        "frozen_public_assets",
    )
    vae = _VAE()
    assets = _assets(LF_CORE_CANDIDATE_ID, vae)
    correct_key = b"0123456789abcdef0123456789abcdef"
    wrong_key = b"abcdef0123456789abcdef0123456789"
    vae.observation = reconstruct_lf_carrier(
        correct_key,
        (1, 4, 32, 32),
        assets,
        dtype=torch.float32,
        device="cpu",
    )
    image = Image.new("RGB", (32, 32), color=(20, 40, 60))

    assert score_lf_image(image, correct_key, assets) == pytest.approx(1.0, abs=1e-6)
    assert score_lf_image(image, correct_key, assets) > score_lf_image(image, wrong_key, assets)
    assert vae.encode_calls == 3
    with pytest.raises(TypeError, match="unexpected keyword"):
        score_lf_image(image, correct_key, assets, prompt="private")


@pytest.mark.unit
def test_blocknorm_radial_blocks_are_disjoint_and_exactly_partition_shell() -> None:
    assets = _blocknorm_assets()
    shell = radial_frequency_mask(128, 128, _spec(assets))
    union = np.zeros_like(shell)
    for minimum, maximum, inclusive in LF_BLOCKNORM_RADIAL_BLOCKS:
        block = radial_frequency_mask(
            128,
            128,
            frequency.FrequencyCarrierSpec(
                domain_prefix="lf",
                carrier_method_id=LF_SHELL_CANDIDATE_ID,
                min_radius=minimum,
                max_radius=maximum,
                max_inclusive=inclusive,
                total_relative_l2=0.012,
            ),
        )
        assert not np.any(union & block)
        union |= block
    assert np.array_equal(union, shell)


@pytest.mark.unit
def test_blocknorm_blind_score_centers_each_channel_block_and_is_keyed() -> None:
    correct_key = b"0123456789abcdef0123456789abcdef"
    wrong_key = b"abcdef0123456789abcdef0123456789"
    vae = _VAE()
    assets = _blocknorm_assets(vae)
    carrier = reconstruct_lf_carrier(
        correct_key,
        (1, 4, 64, 64),
        assets,
        dtype=torch.float32,
        device="cpu",
    )
    channel_offsets = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1) * 17.0
    vae.observation = carrier * 3.5 + channel_offsets
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))

    correct = score_lf_image(image, correct_key, assets)
    wrong = score_lf_image(image, wrong_key, assets)

    assert correct == pytest.approx(1.0, abs=1e-6)
    assert correct > wrong
    assert vae.encode_calls == 2


@pytest.mark.unit
def test_blocknorm_score_fails_closed_on_zero_variance_or_identity_drift() -> None:
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    assets = _blocknorm_assets()
    with pytest.raises(ValueError, match="nonzero block variance"):
        score_lf_image(image, b"0123456789abcdef0123456789abcdef", assets)
    with pytest.raises(ValueError, match="frozen shell carrier"):
        FrozenLFPublicAssets(
            vae=_VAE(),
            image_processor=_Processor(),
            image_processor_id="sd35-vae-image-processor-v1",
            candidate_id=LF_CORE_CANDIDATE_ID,
            detector_statistic_id=LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
            evaluated_candidate_id=LF_BLOCKNORM_EVALUATED_CANDIDATE_ID,
        )


@pytest.mark.unit
def test_blocknorm_equal_weight_median_is_not_dominated_by_one_high_energy_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = b"0123456789abcdef0123456789abcdef"
    assets = _blocknorm_assets()
    carrier = reconstruct_lf_carrier(
        key,
        (1, 4, 64, 64),
        assets,
        dtype=torch.float32,
        device="cpu",
    )
    spectrum = torch.fft.rfft2(carrier, norm="ortho")
    first = LF_BLOCKNORM_RADIAL_BLOCKS[0]
    nuisance_mask = torch.from_numpy(
        radial_frequency_mask(
            64,
            64,
            frequency.FrequencyCarrierSpec(
                domain_prefix="lf",
                carrier_method_id=LF_SHELL_CANDIDATE_ID,
                min_radius=first[0],
                max_radius=first[1],
                max_inclusive=first[2],
                total_relative_l2=0.012,
            ),
        )
    )
    spectrum[0, 0][nuisance_mask] *= -1000.0
    observation = torch.fft.irfft2(spectrum, s=(64, 64), norm="ortho")
    monkeypatch.setattr(frequency, "encode_final_rgb_image", lambda *args: observation)

    score = score_lf_image(Image.new("RGB", (32, 32)), key, assets)

    assert score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
def test_balanced_carrier_is_deterministic_keyed_zero_sum_and_public_shape_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_calls: list[tuple[str, int]] = []
    sign_calls: list[tuple[str, tuple[int, ...], object]] = []
    original_bytes = frequency.prg_bytes
    original_signs = frequency.prg_rademacher

    def traced_bytes(key: object, domain: str, length: int) -> bytes:
        order_calls.append((domain, length))
        return original_bytes(key, domain, length)

    def traced_signs(
        key: object,
        domain: str,
        shape: tuple[int, ...],
        *,
        dtype: object,
    ) -> np.ndarray:
        sign_calls.append((domain, shape, dtype))
        return original_signs(key, domain, shape, dtype=dtype)

    monkeypatch.setattr(frequency, "prg_bytes", traced_bytes)
    monkeypatch.setattr(frequency, "prg_rademacher", traced_signs)
    key = b"0123456789abcdef0123456789abcdef"
    assets = _balanced_assets()
    carrier = reconstruct_lf_carrier(
        key, (1, 4, 64, 64), assets, dtype=torch.float64, device="cpu"
    )
    repeat = reconstruct_lf_carrier(
        key, (1, 4, 64, 64), assets, dtype=torch.float64, device="cpu"
    )
    wrong = reconstruct_lf_carrier(
        b"abcdef0123456789abcdef0123456789",
        (1, 4, 64, 64),
        assets,
        dtype=torch.float64,
        device="cpu",
    )

    assert torch.equal(carrier, repeat)
    assert not torch.equal(carrier, wrong)
    assert torch.linalg.vector_norm(carrier).item() == pytest.approx(1.0, abs=1e-10)
    assert len(order_calls) == len(sign_calls) == 3 * 4 * 4
    for domain, length in order_calls:
        assert domain.endswith("purpose=pair-order")
        assert length % 32 == 0 and length >= 64
        assert LF_BALANCED_BLOCKS_CARRIER_METHOD_ID in domain
        assert any(f"bounds={block[3]}" in domain for block in LF_BALANCED_BLOCKS_RADIAL_BLOCKS)
        assert not any(token in domain for token in ("unit", "prompt", "seed", "evaluated", "winner"))
    for domain, shape, dtype in sign_calls:
        assert domain.endswith("purpose=pair-sign")
        assert len(shape) == 1 and shape[0] >= 1
        assert dtype is np.float32


@pytest.mark.unit
def test_balanced_carrier_fails_closed_for_batch_and_preserves_rfft_pair_support() -> None:
    key = b"0123456789abcdef0123456789abcdef"
    assets = _balanced_assets()
    with pytest.raises(ValueError, match="N=1"):
        reconstruct_lf_carrier(
            key, (2, 4, 64, 64), assets, dtype=torch.float32, device="cpu"
        )
    carrier = reconstruct_lf_carrier(
        key, (1, 4, 64, 64), assets, dtype=torch.float64, device="cpu"
    )
    spectrum = torch.fft.rfft2(carrier, norm="ortho").real
    for channel in range(4):
        assert torch.allclose(
            spectrum[0, channel, :, 0],
            torch.zeros_like(spectrum[0, channel, :, 0]),
            rtol=0.0,
            atol=1e-10,
        )
        assert torch.allclose(
            spectrum[0, channel, :, 32],
            torch.zeros_like(spectrum[0, channel, :, 32]),
            rtol=0.0,
            atol=1e-10,
        )
        for minimum, maximum, inclusive, _ in LF_BALANCED_BLOCKS_RADIAL_BLOCKS:
            block = radial_frequency_mask(
                64,
                64,
                frequency.FrequencyCarrierSpec(
                    domain_prefix="lf",
                    carrier_method_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
                    min_radius=minimum,
                    max_radius=maximum,
                    max_inclusive=inclusive,
                    total_relative_l2=0.012,
                ),
            )
            block[:, 0] = False
            block[:, 32] = False
            values = spectrum[0, channel][torch.from_numpy(block)]
            assert values.sum().item() == pytest.approx(0.0, abs=1e-10)
            assert torch.count_nonzero(values.abs() > 1e-10).item() == 2 * (
                int(block.sum()) // 2
            )


@pytest.mark.unit
def test_balanced_pair_order_collision_uses_lexicographic_coordinates_and_exact_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_calls: list[tuple[str, int]] = []
    sign_calls: list[tuple[str, tuple[int, ...]]] = []

    def collision_tokens(key: object, domain: str, length: int) -> bytes:
        del key
        order_calls.append((domain, length))
        return bytes(length)

    def positive_signs(
        key: object,
        domain: str,
        shape: tuple[int, ...],
        *,
        dtype: object,
    ) -> np.ndarray:
        del key
        assert dtype is np.float32
        sign_calls.append((domain, shape))
        return np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(frequency, "prg_bytes", collision_tokens)
    monkeypatch.setattr(frequency, "prg_rademacher", positive_signs)
    assets = _balanced_assets()
    components = frequency._balanced_block_components(
        b"0123456789abcdef0123456789abcdef",
        (1, 1, 64, 64),
        _spec(assets),
        LF_BALANCED_BLOCKS_RADIAL_BLOCKS,
        device="cpu",
    )
    assert len(components) == 4
    for block_index, ((minimum, maximum, inclusive, token), component) in enumerate(
        zip(LF_BALANCED_BLOCKS_RADIAL_BLOCKS, components)
    ):
        mask = radial_frequency_mask(
            64,
            64,
            frequency.FrequencyCarrierSpec(
                domain_prefix="lf",
                carrier_method_id=LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
                min_radius=minimum,
                max_radius=maximum,
                max_inclusive=inclusive,
                total_relative_l2=0.012,
            ),
        )
        mask[:, 0] = False
        mask[:, 32] = False
        coordinates = [tuple(int(value) for value in coordinate) for coordinate in np.argwhere(mask)]
        spectrum = torch.fft.rfft2(component, norm="ortho").real
        scale = abs(float(spectrum[coordinates[0]].item()))
        for index, coordinate in enumerate(coordinates[: 2 * (len(coordinates) // 2)]):
            expected = scale if index % 2 == 0 else -scale
            assert float(spectrum[coordinate].item()) == pytest.approx(expected, abs=1e-10)
        if len(coordinates) % 2:
            assert float(spectrum[coordinates[-1]].item()) == pytest.approx(0.0, abs=1e-10)
        prefix = (
            f"carrier_method_id={LF_BALANCED_BLOCKS_CARRIER_METHOD_ID}/"
            "channels=1/height=64/width=64/channel=0/"
            f"block={block_index}/bounds={token}/purpose="
        )
        assert order_calls[block_index] == (prefix + "pair-order", 32 * len(coordinates))
        assert sign_calls[block_index] == (prefix + "pair-sign", (len(coordinates) // 2,))


@pytest.mark.unit
def test_balanced_components_have_equal_precast_energy_and_actual_dtype_budget() -> None:
    key = b"0123456789abcdef0123456789abcdef"
    assets = _balanced_assets()
    components = frequency._balanced_block_components(
        key,
        (1, 4, 64, 64),
        _spec(assets),
        LF_BALANCED_BLOCKS_RADIAL_BLOCKS,
        device="cpu",
    )
    assert len(components) == 16
    assert [torch.linalg.vector_norm(component).item() for component in components] == pytest.approx(
        [0.25] * 16, rel=1e-10, abs=1e-10
    )
    latents = torch.linspace(-1.0, 1.0, 4 * 64 * 64, dtype=torch.float16).reshape(
        1, 4, 64, 64
    )
    injected, measurement = inject_lf_carrier(latents, key, assets)
    assert injected.dtype is torch.float16
    assert 0.0 < measurement.relative_l2 <= 0.012


@pytest.mark.unit
def test_balanced_blind_score_reconstructs_same_carrier_from_final_image() -> None:
    key = b"0123456789abcdef0123456789abcdef"
    wrong_key = b"abcdef0123456789abcdef0123456789"
    vae = _VAE()
    assets = _balanced_assets(vae)
    vae.observation = reconstruct_lf_carrier(
        key, (1, 4, 64, 64), assets, dtype=torch.float32, device="cpu"
    )
    image = Image.new("RGB", (64, 64), color=(20, 30, 40))
    correct = score_lf_image(image, key, assets)
    wrong = score_lf_image(image, wrong_key, assets)
    assert correct == pytest.approx(1.0, abs=1e-5)
    assert correct > wrong
    assert vae.encode_calls == 2
