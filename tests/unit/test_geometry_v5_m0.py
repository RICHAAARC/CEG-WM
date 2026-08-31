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
def test_m0_template_initial_z_t_injection_and_similarity_direction_are_pure_math_only() -> None:
    template = method.build_hermitian_x_template()
    assert len(template) == 16
    latent = tuple(tuple(tuple(0.0 for _ in range(8)) for _ in range(8)) for _ in range(4))
    injected = method.inject_initial_z_t_x_template(latent, template)
    assert len(injected) == 4 and any(value != 0.0 for row in injected[3] for value in row)
    spectrum = method._dft2(injected[3])
    point = template[0]
    y, x = method._frequency_bin(point.frequency_y, 8), method._frequency_bin(point.frequency_x, 8)
    assert abs(spectrum[y][x]) > 0.0
    assert spectrum[y][x] == pytest.approx(spectrum[(-y) % 8][(-x) % 8].conjugate())
    assert all(isinstance(value, float) for row in injected[3] for value in row)
    estimate = method.estimate_rotation_scale_from_peak_pairs(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    )
    assert estimate.rotation_degrees == pytest.approx(0.0)
    assert estimate.scale == pytest.approx(2.0)
    H = method.assemble_attacked_to_canonical_similarity(0.0, 2.0, -0.5, -0.5)
    assert H == ((2.0, -0.0, -0.5), (0.0, 2.0, -0.5), (0.0, 0.0, 1.0))


@pytest.mark.unit
def test_recovered_z_t_rotation_scale_search_is_blind_and_fails_closed_on_bad_data() -> None:
    size = 16
    plane = [[0.0 for _ in range(size)] for _ in range(size)]
    for point in method.build_hermitian_x_template():
        y = method._frequency_bin(0.5 * point.frequency_y, size)
        x = method._frequency_bin(0.5 * point.frequency_x, size)
        for row in range(size):
            for column in range(size):
                plane[row][column] += math.cos(2.0 * math.pi * ((y * row / size) + (x * column / size)))
    recovered = [tuple(tuple(0.0 for _ in range(size)) for _ in range(size)) for _ in range(4)]
    recovered[3] = tuple(tuple(row) for row in plane)
    estimate = method.estimate_rotation_scale_from_recovered_z_t(recovered, ((0.0, 1.0), (0.0, 0.5)))
    assert estimate.rotation_degrees == 0.0 and estimate.scale == 0.5
    forward_rotation_degrees = 10.0
    forward_angle = math.radians(forward_rotation_degrees)
    rotated_plane = [[0.0 for _ in range(size)] for _ in range(size)]
    for point in method.build_hermitian_x_template():
        observed_y = 0.5 * (math.sin(forward_angle) * point.frequency_x + math.cos(forward_angle) * point.frequency_y)
        observed_x = 0.5 * (math.cos(forward_angle) * point.frequency_x - math.sin(forward_angle) * point.frequency_y)
        y = method._frequency_bin(observed_y, size)
        x = method._frequency_bin(observed_x, size)
        for row in range(size):
            for column in range(size):
                rotated_plane[row][column] += math.cos(2.0 * math.pi * ((y * row / size) + (x * column / size)))
    recovered[3] = tuple(tuple(row) for row in rotated_plane)
    inverse_estimate = method.estimate_rotation_scale_from_recovered_z_t(
        recovered, ((0.0, 0.5), (10.0, 0.5))
    )
    assert inverse_estimate.rotation_degrees == -10.0 and inverse_estimate.scale == 0.5
    flat = tuple(tuple(tuple(0.0 for _ in range(8)) for _ in range(8)) for _ in range(4))
    with pytest.raises(ValueError, match="usable"):
        method.estimate_rotation_scale_from_recovered_z_t(flat, ((0.0, 1.0),))
    with pytest.raises(ValueError, match="candidate grid"):
        method.estimate_rotation_scale_from_recovered_z_t(recovered, ((True, 1.0),))


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
    latents = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
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
    latents = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
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
    latents = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
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
    latents = torch.zeros((1, 4, 64, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="cast to latent dtype has non-finite"):
        method.inject_initial_z_t_x_template_torch(latents, method.build_hermitian_x_template())
