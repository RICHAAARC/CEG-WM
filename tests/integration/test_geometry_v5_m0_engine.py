from __future__ import annotations

import ast
import inspect
import math

import pytest

from cegwm.protocol.geometry_v5_m0 import GeometryV5M0RawRecord
from cegwm.runtime import geometry_v5_m0_sd21 as runtime
from cegwm.runtime.geometry_v5_m0_sd21 import (
    SD21M0Identity,
    _grid_translation_to_unit_image,
    _unit_image_translation_to_grid,
    diagnostic_rs_candidate_landscape,
    diagnostic_translation_surface_controls,
    estimate_bound_blind_rst,
    invert_bound_sd21_attacked_rgb,
    public_runtime_capabilities,
    recover_and_estimate_bound_sd21,
    recover_and_estimate_from_attacked_rgb,
    run_generation_with_initial_z_t,
)


@pytest.mark.integration
def test_private_coordinate_helpers_are_finite_and_match_the_public_H_basis() -> None:
    unit = 3 / 64
    grid = _unit_image_translation_to_grid(unit)
    assert grid == pytest.approx(6 / 63)
    assert _grid_translation_to_unit_image(grid) == pytest.approx(unit)
    for invalid in (True, float("nan"), float("inf"), "0.1"):
        with pytest.raises(ValueError, match="finite non-bool"):
            _unit_image_translation_to_grid(invalid)
        with pytest.raises(ValueError, match="finite non-bool"):
            _grid_translation_to_unit_image(invalid)


@pytest.mark.integration
def test_production_estimator_signature_and_source_have_no_isolation_inputs() -> None:
    signature = inspect.signature(estimate_bound_blind_rst)
    assert tuple(signature.parameters) == ("recovered_z_t", "identity")
    source = inspect.getsource(estimate_bound_blind_rst)
    function = ast.parse(source).body[0]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert not {"truth", "probe", "probes", "isolation", "controls"} & names
    assert "diagnostic_rs_candidate_landscape" not in source
    assert "diagnostic_translation_surface_controls" not in source


@pytest.mark.integration
def test_diagnostic_helpers_reject_nonexact_shape_and_nonfinite_before_scoring_when_torch_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls = {"score": 0, "resample": 0}

    def score(*_args: object) -> tuple[float, float]:
        calls["score"] += 1
        return 1.0, 1.0

    def resample(*_args: object) -> tuple[object, float]:
        calls["resample"] += 1
        raise AssertionError("shape/finiteness rejection must precede canonicalization")

    monkeypatch.setattr(runtime, "_normalized_template_match_torch", score)
    monkeypatch.setattr(runtime, "_resample_recovered_to_canonical", resample)
    invalid = [
        torch.zeros((2, 4, 64, 64), dtype=torch.float32),
        torch.zeros((1, 3, 64, 64), dtype=torch.float32),
        torch.zeros((1, 4, 32, 64), dtype=torch.float32),
        torch.full((1, 4, 64, 64), float("nan"), dtype=torch.float32),
        torch.full((1, 4, 64, 64), float("inf"), dtype=torch.float32),
    ]
    for recovered in invalid:
        with pytest.raises(ValueError):
            diagnostic_rs_candidate_landscape(recovered)
        with pytest.raises(ValueError):
            diagnostic_translation_surface_controls(recovered, {"control": {"rotation_degrees": 0.0, "scale": 1.0}})
    assert calls == {"score": 0, "resample": 0}


@pytest.mark.integration
def test_diagnostic_rs_top_k_and_probe_scores_are_deterministic_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from cegwm.method.geometry_v5_m0 import build_hermitian_x_template, inject_initial_z_t_x_template_torch

    torch.manual_seed(7512)
    recovered = inject_initial_z_t_x_template_torch(
        torch.randn((1, 4, 64, 64), dtype=torch.float32), build_hermitian_x_template(),
    )
    probes = {
        "expected_forward": {"forward_rotation_degrees": 10.0, "scale": 1.0},
        "mirror": {"forward_rotation_degrees": -10.0, "scale": 1.0},
    }
    first = diagnostic_rs_candidate_landscape(recovered, probes, top_k=5)
    second = diagnostic_rs_candidate_landscape(recovered, probes, top_k=5)
    assert first == second and first["grid_size"] == 961 and len(first["top_k"]) == 5
    assert [item["name"] for item in first["probe_candidates"]] == ["expected_forward", "mirror"]
    assert first["probe_candidates"][0]["attacked_to_canonical_rotation_degrees"] == -10.0
    assert all(set(item) == {"forward_rotation_degrees", "attacked_to_canonical_rotation_degrees", "scale", "score", "correlation", "local_contrast"} for item in first["top_k"])


@pytest.mark.integration
def test_translation_surface_controls_are_deterministic_and_do_not_change_raw_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from cegwm.method.geometry_v5_m0 import build_hermitian_x_template, inject_initial_z_t_x_template_torch

    torch.manual_seed(7513)
    recovered = inject_initial_z_t_x_template_torch(
        torch.randn((1, 4, 64, 64), dtype=torch.float32), build_hermitian_x_template(),
    )
    raw_before = estimate_bound_blind_rst(recovered)
    controls = {
        "frozen_raw": {"rotation_degrees": float(raw_before.rotation_degrees), "scale": float(raw_before.scale)},
        "parsed_inverse_truth": {"rotation_degrees": 0.0, "scale": 1.0},
    }
    first = diagnostic_translation_surface_controls(recovered, controls, top_k=5)
    second = diagnostic_translation_surface_controls(recovered, controls, top_k=5)
    raw_after = estimate_bound_blind_rst(recovered)
    assert first == second and raw_after == raw_before
    assert [item["name"] for item in first["controls"]] == ["frozen_raw", "parsed_inverse_truth"]
    assert all(len(item["top_k_nonlocal_peaks"]) == 5 for item in first["controls"])
    assert all("phase_psr" in item and "zero_shift_score" in item and "ambiguity_diagnostics" in item for item in first["controls"])


@pytest.mark.integration
def test_injected_fake_adapters_exercise_only_fixed_boundaries_not_real_mechanism() -> None:
    generated: dict[str, object] = {}

    def fake_generator(**kwargs: object) -> str:
        generated.update(kwargs)
        return "fake-final-rgb"

    assert run_generation_with_initial_z_t(fake_generator, "fake-z", "manifest prompt") == "fake-final-rgb"
    assert generated["prompt"] == "manifest prompt" and generated["num_inference_steps"] == 50 and generated["guidance_scale"] == 7.5

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
def test_concrete_estimator_rejects_batch_before_fft_or_candidate_scoring_when_torch_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls = {"fft2": 0, "candidate_score": 0}
    original_fft2 = torch.fft.fft2
    original_candidate_score = runtime._normalized_template_match_torch

    def counted_fft2(*args: object, **kwargs: object) -> object:
        calls["fft2"] += 1
        return original_fft2(*args, **kwargs)

    def counted_candidate_score(*args: object, **kwargs: object) -> tuple[float, float]:
        calls["candidate_score"] += 1
        return original_candidate_score(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "fft2", counted_fft2)
    monkeypatch.setattr(runtime, "_normalized_template_match_torch", counted_candidate_score)
    recovered = torch.randn((2, 4, 64, 64), dtype=torch.float32)

    assert estimate_bound_blind_rst(recovered).status.value == "FAILED"
    assert calls == {"fft2": 0, "candidate_score": 0}


@pytest.mark.integration
def test_concrete_estimator_returns_signed_translation_after_identity_normalization_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    from cegwm.method.geometry_v5_m0 import build_hermitian_x_template, inject_initial_z_t_x_template_torch

    torch.manual_seed(7510)
    canonical = inject_initial_z_t_x_template_torch(
        torch.randn((1, 4, 64, 64), dtype=torch.float32), build_hermitian_x_template(),
    )
    # g(x)=f(x+u): unit-image H translation is converted once for the grid.
    unit_u = (3 / 64, 4 / 64)
    coordinates = torch.linspace(-1.0, 1.0, 64)
    observed_y, observed_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    canonical_x = observed_x + _unit_image_translation_to_grid(unit_u[0])
    canonical_y = observed_y + _unit_image_translation_to_grid(unit_u[1])
    grid = torch.stack((canonical_x, canonical_y), dim=-1).unsqueeze(0)
    recovered = canonical.clone()
    recovered[:, 3] = functional.grid_sample(
        canonical[:, 3].unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, 0]
    raw = estimate_bound_blind_rst(recovered)
    assert raw.status.value == "ESTIMATE_AVAILABLE"
    assert raw.rotation_degrees == pytest.approx(0.0)
    assert raw.scale == pytest.approx(1.0)
    assert raw.tx == pytest.approx(unit_u[0])
    assert raw.ty == pytest.approx(unit_u[1])


@pytest.mark.integration
def test_known_latent_rotation_scale_direction_is_attacked_to_canonical_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    from cegwm.method.geometry_v5_m0 import build_hermitian_x_template, inject_initial_z_t_x_template_torch

    scale, phi = 1.0 / 1.1, 10.0
    rotation = -phi
    forward_t = (0.08, 0.0)
    angle = math.radians(rotation)
    cosine, sine = math.cos(angle), math.sin(angle)
    expected_u = (
        -scale * (cosine * forward_t[0] - sine * forward_t[1]),
        -scale * (sine * forward_t[0] + cosine * forward_t[1]),
    )
    torch.manual_seed(7511)
    canonical = inject_initial_z_t_x_template_torch(
        torch.randn((1, 4, 64, 64), dtype=torch.float32), build_hermitian_x_template(),
    )
    coordinates = torch.linspace(-1.0, 1.0, 64)
    observed_y, observed_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    canonical_x = scale * (cosine * observed_x - sine * observed_y) + _unit_image_translation_to_grid(expected_u[0])
    canonical_y = scale * (sine * observed_x + cosine * observed_y) + _unit_image_translation_to_grid(expected_u[1])
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
            self.timesteps = list(range(50))
            self.alphas_cumprod = torch.linspace(0.99, 0.50, 50, dtype=torch.float32)
            self.calls = 0

        def set_timesteps(self, _steps: int, **_kwargs: object) -> None:
            self.calls += 1

    pipeline = type("Pipeline", (), {})()
    pipeline.device = torch.device("cpu")
    pipeline.vae, pipeline.tokenizer, pipeline.text_encoder = VAE(), Tokenizer(), TextEncoder()
    pipeline.unet, pipeline.scheduler = UNet(), Scheduler()
    for bad_identity in (
        SD21M0Identity(steps=1),
        SD21M0Identity(steps=50.0),
        SD21M0Identity(steps=True),
        SD21M0Identity(eta=0.1),
        SD21M0Identity(eta=False),
        SD21M0Identity(model_revision="0" * 40),
        SD21M0Identity(inversion_guidance_scale=2.0),
        SD21M0Identity(guidance_scale=7),
        SD21M0Identity(inversion_guidance_scale=1),
    ):
        with pytest.raises(ValueError, match="frozen SD2.1"):
            invert_bound_sd21_attacked_rgb(pipeline, image, bad_identity)
        assert pipeline.scheduler.calls == 0 and pipeline.vae.calls == 0 and pipeline.unet.calls == 0
        assert recover_and_estimate_bound_sd21(pipeline, image, bad_identity).status.value == "FAILED"
        assert pipeline.scheduler.calls == 0 and pipeline.vae.calls == 0 and pipeline.unet.calls == 0
    inverted = invert_bound_sd21_attacked_rgb(pipeline, image)
    assert inverted.shape == (1, 4, 64, 64)
    assert pipeline.vae.calls == 1 and pipeline.unet.calls == 49 and pipeline.scheduler.calls == 1
    raw = recover_and_estimate_bound_sd21(pipeline, image)
    assert raw.status.value == "FAILED"
