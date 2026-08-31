from __future__ import annotations

import math

import pytest

from cegwm.protocol.geometry_v5_m0 import GeometryV5M0RawRecord
from cegwm.runtime import geometry_v5_m0_sd21 as runtime
from cegwm.runtime.geometry_v5_m0_sd21 import (
    estimate_bound_blind_rst,
    invert_bound_sd21_attacked_rgb,
    public_runtime_capabilities,
    recover_and_estimate_bound_sd21,
    recover_and_estimate_from_attacked_rgb,
    run_generation_with_initial_z_t,
)


@pytest.mark.integration
def test_injected_fake_adapters_exercise_only_fixed_boundaries_not_real_mechanism() -> None:
    generated: dict[str, object] = {}

    def fake_generator(**kwargs: object) -> str:
        generated.update(kwargs)
        return "fake-final-rgb"

    assert run_generation_with_initial_z_t(fake_generator, "fake-z", "manifest prompt") == "fake-final-rgb"
    assert generated["prompt"] == "manifest prompt" and generated["num_inference_steps"] == 50

    def fake_inverter(image: object, **kwargs: object) -> str:
        assert image == "attacked-rgb" and kwargs["prompt"] == "" and kwargs["guidance_scale"] == 1.0
        return "fake-recovered-z"

    def fake_estimator(recovered: object) -> GeometryV5M0RawRecord:
        assert recovered == "fake-recovered-z"
        return GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})

    raw = recover_and_estimate_from_attacked_rgb("attacked-rgb", fake_inverter, fake_estimator)
    assert raw.status.value == "FAILED"
    capabilities = public_runtime_capabilities()
    assert capabilities["real_model_adapter_bound"] is True
    assert capabilities["fake_injected_adapter_is_real_evidence"] is False


@pytest.mark.integration
def test_concrete_estimator_fails_closed_for_flat_or_ambiguous_spectra_when_torch_available(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    flat = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    assert estimate_bound_blind_rst(flat).status.value == "FAILED"
    recovered = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    recovered[:, 3] = torch.randn((1, 64, 64), dtype=torch.float32)
    monkeypatch.setattr(runtime, "_bilinear_periodic", lambda *_args: 1.0)
    assert estimate_bound_blind_rst(recovered).status.value == "FAILED"


@pytest.mark.integration
def test_concrete_estimator_returns_signed_translation_after_identity_normalization_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from cegwm.method.geometry_v5_m0 import build_hermitian_x_template, inject_initial_z_t_x_template_torch

    canonical = inject_initial_z_t_x_template_torch(
        torch.zeros((1, 4, 64, 64), dtype=torch.float32), build_hermitian_x_template(),
    )
    # g(x)=f(x+u): a right/down canonical u appears as a negative tensor roll.
    recovered = canonical.clone()
    recovered[:, 3] = torch.roll(canonical[:, 3], shifts=(-4, -3), dims=(-2, -1))
    raw = estimate_bound_blind_rst(recovered)
    assert raw.status.value == "ESTIMATE_AVAILABLE"
    assert raw.rotation_degrees == pytest.approx(0.0)
    assert raw.scale == pytest.approx(1.0)
    assert raw.tx == pytest.approx(3 / 64)
    assert raw.ty == pytest.approx(4 / 64)


@pytest.mark.integration
def test_concrete_estimator_uses_B_times_forward_translation_in_compound_fixture_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    from cegwm.method.geometry_v5_m0 import build_hermitian_x_template, inject_initial_z_t_x_template_torch

    scale, phi = 0.93, 7.0
    rotation = -phi
    forward_t = (0.05, -0.04)
    angle = math.radians(rotation)
    cosine, sine = math.cos(angle), math.sin(angle)
    expected_u = (
        -scale * (cosine * forward_t[0] - sine * forward_t[1]),
        -scale * (sine * forward_t[0] + cosine * forward_t[1]),
    )
    canonical = inject_initial_z_t_x_template_torch(
        torch.zeros((1, 4, 64, 64), dtype=torch.float32), build_hermitian_x_template(),
    )
    coordinates = torch.linspace(-1.0, 1.0, 64)
    observed_y, observed_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    canonical_x = scale * (cosine * observed_x - sine * observed_y) + expected_u[0]
    canonical_y = scale * (sine * observed_x + cosine * observed_y) + expected_u[1]
    grid = torch.stack((canonical_x, canonical_y), dim=-1).unsqueeze(0)
    recovered = canonical.clone()
    recovered[:, 3] = functional.grid_sample(
        canonical[:, 3].unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, 0]
    raw = estimate_bound_blind_rst(recovered)
    assert raw.status.value == "ESTIMATE_AVAILABLE"
    assert raw.rotation_degrees == pytest.approx(rotation)
    assert raw.scale == pytest.approx(scale)
    assert raw.tx == pytest.approx(expected_u[0], abs=1 / 64)
    assert raw.ty == pytest.approx(expected_u[1], abs=1 / 64)
    assert raw.diagnostics["canonical_overlap"] > 0.5


@pytest.mark.integration
def test_concrete_combined_entry_exercises_fake_vae_ddim_components_when_available() -> None:
    torch = pytest.importorskip("torch")
    image = pytest.importorskip("PIL.Image").new("RGB", (512, 512), color=(32, 64, 96))

    class Distribution:
        def mode(self) -> object:
            return torch.zeros((1, 4, 64, 64), dtype=torch.float32)

    class VAE:
        config = type("Config", (), {"scaling_factor": 1.0})()

        def __init__(self) -> None:
            self.calls = 0

        def encode(self, pixels: object) -> object:
            self.calls += 1
            return type("Encoded", (), {"latent_dist": Distribution()})()

    class Tokenizer:
        model_max_length = 3

        def __call__(self, *_args: object, **_kwargs: object) -> object:
            return type("Tokens", (), {"input_ids": torch.zeros((1, 3), dtype=torch.long)})()

    class TextEncoder:
        def __call__(self, ids: object) -> tuple[object]:
            return (torch.zeros((1, 3, 2), dtype=torch.float32),)

    class UNet:
        dtype = torch.float32

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, latent: object, _timestep: object, **_kwargs: object) -> object:
            self.calls += 1
            return type("Output", (), {"sample": torch.zeros_like(latent)})()

    class Scheduler:
        def __init__(self) -> None:
            self.timesteps = [2, 1, 0]
            self.alphas_cumprod = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
            self.calls = 0

        def set_timesteps(self, _steps: int, **_kwargs: object) -> None:
            self.calls += 1

    pipeline = type("Pipeline", (), {})()
    pipeline.device = torch.device("cpu")
    pipeline.vae, pipeline.tokenizer, pipeline.text_encoder = VAE(), Tokenizer(), TextEncoder()
    pipeline.unet, pipeline.scheduler = UNet(), Scheduler()
    inverted = invert_bound_sd21_attacked_rgb(pipeline, image)
    assert inverted.shape == (1, 4, 64, 64)
    assert pipeline.vae.calls == 1 and pipeline.unet.calls == 2 and pipeline.scheduler.calls == 1
    raw = recover_and_estimate_bound_sd21(pipeline, image)
    assert raw.status.value == "FAILED"
