from __future__ import annotations

import hashlib
import json
import math
import ast
from pathlib import Path

import pytest

from cegwm.method import geometry_v5_m0 as method
from cegwm.protocol import geometry_v5_m0 as protocol


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_m0_byte_bindings_sources_roster_and_engineering_ceiling_are_frozen() -> None:
    contract = protocol.load_geometry_v5_m0_contract(_ROOT)
    assert hashlib.sha256((_ROOT / protocol.M0_CONFIG_PATH).read_bytes()).hexdigest() == protocol.M0_CONFIG_SHA256
    assert hashlib.sha256((_ROOT / protocol.M0_MANIFEST_PATH).read_bytes()).hexdigest() == protocol.M0_MANIFEST_SHA256
    assert tuple(unit.seed for unit in contract.units) == (7501, 7502, 7503, 7504)
    assert all(unit.seed not in {6201, 6202, 6203, 6204} for unit in contract.units)
    assert len(contract.units) * len(contract.config["development"]["attacks"]) == 44
    assert [attack["attack_id"] for attack in contract.config["development"]["attacks"]] == [
        "identity", "rotation_-10", "rotation_+10", "scale_0.9", "scale_1.1",
        "translation_x_-0.08", "translation_x_+0.08", "translation_y_-0.08",
        "translation_y_+0.08", "compound_rot+7_scale0.93_tx+0.05_ty-0.04",
        "compound_rot-7_scale1.07_tx-0.05_ty+0.04",
    ]
    assert contract.config["source_bindings"]["maxsive"]["exact"] == "a9554024aed176e705cc15ca1cbd31b9c7f75bfb"
    assert contract.config["source_bindings"]["tree_ring"]["exact"] == "3015283d9cf82e90b628f02ad2121bd37408ca9a"
    assert contract.config["engineering_evaluation"]["claim_ceiling"] == protocol.M0_CLAIM_CEILING


@pytest.mark.unit
def test_m0_direct_injected_z_t_writer_detector_closure_uses_relative_setting_only() -> None:
    template = method.build_hermitian_x_template()
    assert len(template) == 16
    assert method.M0_OFFICIAL_X_ANGLES_DEGREES == (1.0, 135.0)
    size = 16
    latent = tuple(
        tuple(tuple(math.sin((channel + 1) * (row + 1) * (column + 2)) for column in range(size)) for row in range(size))
        for channel in range(4)
    )
    before_spectrum = method._dft2(latent[3])
    target = method._relative_coefficient_target(before_spectrum)
    injected = method.inject_initial_z_t_x_template(latent, template)
    assert len(injected) == 4 and any(value != 0.0 for row in injected[3] for value in row)
    spectrum = method._dft2(injected[3])
    pairs = method._template_bin_pairs(template, size, size)
    for (y, x), (conjugate_y, conjugate_x) in pairs:
        assert spectrum[y][x] == pytest.approx(target)
        assert spectrum[conjugate_y][conjugate_x] == pytest.approx(target)
    estimate = method.estimate_rotation_scale_from_recovered_z_t(
        injected, ((0.0, 1.0), (10.0, 1.1), (-10.0, 0.9)),
    )
    assert estimate.rotation_degrees == pytest.approx(0.0) and estimate.scale == pytest.approx(1.0)
    assert estimate.diagnostics["normalized_template_correlation"] > 0.0
    assert estimate.diagnostics["nms_psr"] > 1.0
    assert all(isinstance(value, float) for row in injected[3] for value in row)
    paired = method.estimate_rotation_scale_from_peak_pairs(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    )
    assert paired.rotation_degrees == pytest.approx(0.0)
    assert paired.scale == pytest.approx(2.0)
    H = method.assemble_attacked_to_canonical_similarity(0.0, 2.0, -0.5, -0.5)
    assert H == ((2.0, -0.0, -0.5), (0.0, 2.0, -0.5), (0.0, 0.0, 1.0))


@pytest.mark.unit
def test_known_latent_rotation_scale_direction_is_attacked_to_canonical_and_blind() -> None:
    size, forward_rotation_degrees, attacked_to_canonical_scale = 32, 10.0, 1.0 / 1.1
    spectrum = [[0j for _ in range(size)] for _ in range(size)]
    angle = math.radians(forward_rotation_degrees)
    for point in method.build_hermitian_x_template():
        observed_x = attacked_to_canonical_scale * (math.cos(angle) * point.frequency_x - math.sin(angle) * point.frequency_y)
        observed_y = attacked_to_canonical_scale * (math.sin(angle) * point.frequency_x + math.cos(angle) * point.frequency_y)
        if not (-0.5 <= observed_x <= 0.5 and -0.5 <= observed_y <= 0.5):
            continue
        y, x = method._frequency_bin(observed_y, size), method._frequency_bin(observed_x, size)
        spectrum[y][x] = 100.0 + 0j
    plane = tuple(tuple(value.real for value in row) for row in method._idft2(spectrum))
    recovered = tuple(plane if channel == 3 else tuple(tuple(0.0 for _ in range(size)) for _ in range(size)) for channel in range(4))
    estimate = method.estimate_rotation_scale_from_recovered_z_t(
        recovered, ((0.0, 1.0), (10.0, attacked_to_canonical_scale), (8.0, attacked_to_canonical_scale), (10.0, 1.0)),
    )
    assert estimate.rotation_degrees == pytest.approx(-10.0)
    assert estimate.scale == pytest.approx(attacked_to_canonical_scale)
    assert estimate.diagnostics["nms_runner_up_score"] < estimate.score
    flat = tuple(tuple(tuple(0.0 for _ in range(8)) for _ in range(8)) for _ in range(4))
    with pytest.raises(ValueError, match="usable"):
        method.estimate_rotation_scale_from_recovered_z_t(flat, ((0.0, 1.0),))
    with pytest.raises(ValueError, match="candidate grid"):
        method.estimate_rotation_scale_from_recovered_z_t(recovered, ((True, 1.0),))


@pytest.mark.unit
def test_known_latent_translation_uses_masked_template_cross_power_at_one_over_64() -> None:
    size = 64
    spectrum = [[0j for _ in range(size)] for _ in range(size)]
    for y, x in method._template_support(method.build_hermitian_x_template(), size, size):
        spectrum[y][x] = 1.0 + 0j
    canonical = tuple(tuple(value.real for value in row) for row in method._idft2(spectrum))
    shift_x, shift_y = 5, -3
    observed = tuple(
        tuple(canonical[(row + shift_y) % size][(column + shift_x) % size] for column in range(size))
        for row in range(size)
    )
    tx, ty = method.estimate_translation_phase_correlation(canonical, observed)
    assert tx == pytest.approx(shift_x / size)
    assert ty == pytest.approx(shift_y / size)


@pytest.mark.unit
def test_m0_raw_output_has_no_reliable_rectification_or_fabricated_failure() -> None:
    failed = protocol.GeometryV5M0RawRecord("FAILED", None, None, None, None, None, {})
    assert failed.status is protocol.M0RawStatus.FAILED
    with pytest.raises(ValueError, match="fabricate"):
        protocol.GeometryV5M0RawRecord("FAILED", 0.0, None, None, None, None, {})
    available = protocol.GeometryV5M0RawRecord(
        "ESTIMATE_AVAILABLE", 0.0, 1.0, 0.0, 0.0,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), {"phase_peak": 1.0},
    )
    assert available.status is protocol.M0RawStatus.ESTIMATE_AVAILABLE
    scope = protocol.load_geometry_v5_m0_contract(_ROOT).config["scope"]
    assert scope["may_emit_RELIABLE"] is False and scope["may_rectify"] is False and scope["may_vote_content"] is False


@pytest.mark.unit
def test_m0_raw_H_is_exact_nontrivial_attacked_to_canonical_similarity_from_its_rst() -> None:
    H = method.assemble_attacked_to_canonical_similarity(20.0, 1.2, -0.1, 0.05)
    record = protocol.GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", 20.0, 1.2, -0.1, 0.05, H, {})
    assert record.H_hat == H
    with pytest.raises(ValueError, match="positive-scale similarity"):
        protocol.GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", 20.0, 1.2, -0.1, 0.05, ((1.2, 0.2, -0.1), (0.0, 1.2, 0.05), (0.0, 0.0, 1.0)), {})
    with pytest.raises(ValueError, match="positive-scale similarity"):
        protocol.GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", 20.0, 1.2, -0.1, 0.05, ((-1.2, 0.0, -0.1), (0.0, 1.2, 0.05), (0.0, 0.0, 1.0)), {})
    with pytest.raises(ValueError, match="scale must be positive"):
        protocol.GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", 0.0, 0.0, 0.0, 0.0, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), {})
    with pytest.raises(ValueError, match="scale must be positive"):
        protocol.GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", 0.0, -1.0, 0.0, 0.0, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), {})
    with pytest.raises(ValueError, match="match"):
        protocol.GeometryV5M0RawRecord("ESTIMATE_AVAILABLE", 20.0, 1.2, -0.1, 0.05, method.assemble_attacked_to_canonical_similarity(-20.0, 1.0 / 1.2, 0.1, -0.05), {})


@pytest.mark.unit
def test_m0_contract_rejects_noncanonical_bytes_and_forbidden_detector_inputs_are_recorded() -> None:
    raw = (_ROOT / protocol.M0_CONFIG_PATH).read_bytes()
    assert raw == protocol.canonical_json_bytes(json.loads(raw))
    forbidden = protocol.load_geometry_v5_m0_contract(_ROOT).config["scope"]["detector_forbidden_inputs"]
    assert set(forbidden) >= {"original_prompt", "original_z_T", "clean_RGB", "true_H", "evaluation_truth"}


@pytest.mark.unit
def test_fourier_spatial_duality_keeps_spectral_scale_and_inverts_only_rotation() -> None:
    source = (_ROOT / "src/cegwm/method/geometry_v5_m0.py").read_text(encoding="utf-8")
    assert "k_observed = c R(phi) k_canonical" in source
    assert "_normalize_degrees(-forward_rotation_degrees), forward_scale" in source
    assert "1.0 / forward_scale" not in source


@pytest.mark.unit
def test_blind_detector_signature_cannot_receive_clean_or_truth_inputs() -> None:
    signature = method.estimate_rotation_scale_from_recovered_z_t.__annotations__
    assert set(signature) == {"recovered_z_t", "candidate_grid", "return"}
    source = (_ROOT / "src/cegwm/method/geometry_v5_m0.py").read_text(encoding="utf-8")
    detector_source = source[source.index("def estimate_rotation_scale_from_recovered_z_t"):source.index("def estimate_translation_phase_correlation")]
    for forbidden in ("original_z_T", "clean_RGB", "true_H", "evaluation_truth", "prompt"):
        assert forbidden not in detector_source
    assert "normalized_template" in detector_source and "nms_runner_up_score" in detector_source


@pytest.mark.unit
def test_torch_production_boundary_is_lazy_and_validates_template_before_fft_write() -> None:
    source = (_ROOT / "src/cegwm/method/geometry_v5_m0.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert "torch" not in imports
    assert "torch template entries must be XTemplatePoint" in source
    assert "torch Hermitian inverse has non-real residual" in source


@pytest.mark.unit
def test_torch_template_injection_rejects_bad_entries_and_nonfinite_weights_when_available() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(6032)
    latents = torch.randn((1, 4, 64, 64), dtype=torch.float32)
    template = method.build_hermitian_x_template()
    injected = method.inject_initial_z_t_x_template_torch(latents, template)
    assert torch.isfinite(injected).all()
    with pytest.raises(TypeError, match="XTemplatePoint"):
        method.inject_initial_z_t_x_template_torch(latents, (object(),))
    with pytest.raises(ValueError, match="finite"):
        method.inject_initial_z_t_x_template_torch(
            latents, (method.XTemplatePoint(0.2, 0.2, float("nan")),),
        )


@pytest.mark.unit
def test_torch_template_injection_rejects_batch_before_fft_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    calls = {"fft2": 0}
    original_fft2 = torch.fft.fft2

    def counted_fft2(*args: object, **kwargs: object) -> object:
        calls["fft2"] += 1
        return original_fft2(*args, **kwargs)

    monkeypatch.setattr(torch.fft, "fft2", counted_fft2)
    latents = torch.randn((2, 4, 64, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="1x4x64x64"):
        method.inject_initial_z_t_x_template_torch(latents, method.build_hermitian_x_template())

    assert calls == {"fft2": 0}


@pytest.mark.unit
@pytest.mark.parametrize("dtype_name", ("float32", "float16"))
def test_torch_template_injection_accepts_seeded_random_cpu_latent_when_available(dtype_name: str) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(6033)
    dtype = getattr(torch, dtype_name)
    latents = torch.randn((1, 4, 64, 64), dtype=dtype)

    injected = method.inject_initial_z_t_x_template_torch(latents, method.build_hermitian_x_template())

    assert injected.dtype is dtype and injected.device.type == "cpu"
    assert torch.isfinite(injected).all()
    assert torch.equal(injected[:, :3], latents[:, :3])
    assert not torch.equal(injected[:, 3], latents[:, 3])


@pytest.mark.unit
def test_torch_cuda_float16_is_an_availability_skip_only() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA float16 is unavailable")
    assert torch.float16 is not None


@pytest.mark.unit
def test_torch_template_injection_rejects_residual_clearly_above_dtype_tolerance_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original_ifft2 = torch.fft.ifft2

    def nonreal_ifft2(*args: object, **kwargs: object) -> object:
        spatial = original_ifft2(*args, **kwargs)
        residual = method._torch_hermitian_residual_tolerance(spatial, torch) + 1.0
        return torch.complex(spatial.real, torch.full_like(spatial.real, residual))

    monkeypatch.setattr(torch.fft, "ifft2", nonreal_ifft2)
    torch.manual_seed(6034)
    latents = torch.randn((1, 4, 64, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="non-real residual"):
        method.inject_initial_z_t_x_template_torch(latents, method.build_hermitian_x_template())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("component", "nonfinite"),
    (("real", float("nan")), ("real", float("inf")), ("imag", float("nan")), ("imag", float("inf"))),
)
def test_torch_template_injection_rejects_nonfinite_ifft_components_when_available(
    monkeypatch: pytest.MonkeyPatch, component: str, nonfinite: float,
) -> None:
    torch = pytest.importorskip("torch")
    original_ifft2 = torch.fft.ifft2

    def nonfinite_ifft2(*args: object, **kwargs: object) -> object:
        spatial = original_ifft2(*args, **kwargs)
        real, imag = spatial.real.clone(), spatial.imag.clone()
        target = real if component == "real" else imag
        target[..., 0, 0] = nonfinite
        return torch.complex(real, imag)

    monkeypatch.setattr(torch.fft, "ifft2", nonfinite_ifft2)
    torch.manual_seed(6035)
    latents = torch.randn((1, 4, 64, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="non-finite spatial components"):
        method.inject_initial_z_t_x_template_torch(latents, method.build_hermitian_x_template())


@pytest.mark.unit
def test_torch_template_injection_rejects_finite_ifft_that_overflows_latent_dtype_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original_ifft2 = torch.fft.ifft2

    def overflow_ifft2(*args: object, **kwargs: object) -> object:
        spatial = original_ifft2(*args, **kwargs)
        finite_real = torch.full_like(spatial.real, torch.finfo(torch.float16).max * 2.0)
        return torch.complex(finite_real, torch.zeros_like(spatial.real))

    monkeypatch.setattr(torch.fft, "ifft2", overflow_ifft2)
    torch.manual_seed(6036)
    latents = torch.randn((1, 4, 64, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="cast to latent dtype has non-finite"):
        method.inject_initial_z_t_x_template_torch(latents, method.build_hermitian_x_template())
