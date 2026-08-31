from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method import geometry_v4_g1r as method
from cegwm.protocol.geometry_v4_g1r import (
    CONFIG_SHA256,
    DECODER_DTYPE_GUARD_EPS_MULTIPLIER,
    DEVELOPMENT_ARTIFACT_FILES,
    DEVELOPMENT_NOTEBOOK_ID,
    ENERGY_SHARES,
    FIT_GATES,
    FIT_PATCH_WINDOW_DIVISOR,
    FIT_TILE_IDS,
    HOLDOUT_GATES,
    HOLDOUT_FREQUENCY_RADIUS,
    HOLDOUT_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION,
    HOLDOUT_PATCH_WINDOW_DIVISORS,
    LOCAL_PREPROCESSING,
    RGB_CHANNEL_PEAK_CAP,
    RGB_CHANNEL_RMS_CAP,
    SEARCH_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION,
    SPARSE_CHIP_RADIUS_FRACTION,
    SPARSE_DOMAIN_SUPPORT_GRID,
    SPARSE_LOCAL_ACTIVE_MODULUS,
    SPARSE_LOCAL_GRID,
    SPARSE_SEARCH_ACTIVE_MODULUS,
    SPARSE_SEARCH_GRIDS,
    SPARSE_SEARCH_GROUPS,
    SPARSE_SUPPORT_FRACTION,
    TRANSLATION_PSR_MIN,
    TRANSLATION_NMS_RADIUS_PIXELS,
    TRANSLATION_PEAKS_PER_RS,
    VALIDATE_TILE_IDS,
    WRITER_TARGET_RMS_FRACTION,
    contract_sha256,
    derive_g1r_keys,
    load_contract,
    require_split,
)
from cegwm.runtime.geometry_v4_g1r_sd35 import G1RDecoderOutputHook, run_g1r_sd35_pair

ROOT = Path(__file__).resolve().parents[2]
KEY = b"0123456789abcdef"


class _Decoder(torch.nn.Module):
    def forward(self, value):
        return value


class _VAE:
    def __init__(self) -> None:
        self.decoder = _Decoder()


class _Pipeline:
    def __init__(self) -> None:
        self.vae = _VAE()
        self.calls = 0
        self.fail_on_call = None

    def __call__(self, **kwargs):
        self.calls += 1
        decoded = self.vae.decoder(torch.zeros((1, 3, kwargs["height"], kwargs["width"]), dtype=torch.float32))
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic marked generation failure")
        rgb = ((decoded[0].permute(1, 2, 0) + 1.0) / 2.0).clamp(0.0, 1.0)
        image = Image.fromarray((rgb.numpy() * 255.0).round().astype(np.uint8), mode="RGB")
        return SimpleNamespace(images=[image])


@pytest.mark.unit
def test_contract_freezes_domains_tiles_rosters_and_old_seed_rejection() -> None:
    contract = load_contract(ROOT)
    assert contract_sha256(ROOT) == CONFIG_SHA256
    assert ENERGY_SHARES == (.4, .36, .24) and sum(ENERGY_SHARES) == 1.0
    assert WRITER_TARGET_RMS_FRACTION == .25
    assert TRANSLATION_PEAKS_PER_RS == 3 and TRANSLATION_NMS_RADIUS_PIXELS == 2
    assert SEARCH_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION == .1
    assert SPARSE_SEARCH_GRIDS == (8, 12, 16) and SPARSE_SEARCH_GROUPS == 4
    assert SPARSE_SEARCH_ACTIVE_MODULUS == SPARSE_LOCAL_ACTIVE_MODULUS == 2
    assert SPARSE_LOCAL_GRID == SPARSE_DOMAIN_SUPPORT_GRID == 8
    assert SPARSE_CHIP_RADIUS_FRACTION == .2 and SPARSE_SUPPORT_FRACTION == .18
    assert FIT_PATCH_WINDOW_DIVISOR == 20
    assert HOLDOUT_PATCH_WINDOW_DIVISORS == (20, 24) and HOLDOUT_FREQUENCY_RADIUS == (12.0, 31.0)
    assert HOLDOUT_KEYED_FREQUENCY_SUPPORT_MIN_FRACTION == .1
    assert set(FIT_TILE_IDS) | set(VALIDATE_TILE_IDS) == set(range(16))
    assert not set(FIT_TILE_IDS) & set(VALIDATE_TILE_IDS)
    assert len(require_split(contract, "development")) == len(require_split(contract, "confirmation")) == 20
    with pytest.raises(ValueError, match="split"):
        require_split(contract, "mixed")


@pytest.mark.unit
def test_key_domains_and_anchor_energy_are_separate_and_deterministic() -> None:
    keys = derive_g1r_keys(KEY)
    assert len(set(keys.values())) == 3
    for shape in ((64, 64), (128, 128), (512, 512)):
        for key in (KEY, b"separate-key-0001", b"separate-key-0002"):
            fields = method.g1r_anchor_fields(shape, key)
            repeated = method.g1r_anchor_fields(shape, key)
            assert np.array_equal(fields.combined, repeated.combined)
            domains = (fields.search, fields.fit, fields.validate)
            gram = np.asarray([[np.sum(left * right) for right in domains] for left in domains])
            assert gram == pytest.approx(np.eye(3), abs=2e-14)
            assert np.linalg.norm(fields.combined) == pytest.approx(1.0, abs=2e-14)
            components = tuple(np.sqrt(share) * field for share, field in zip(ENERGY_SHARES, domains, strict=True))
            assert tuple(float(np.sum(component * component)) for component in components) == pytest.approx(ENERGY_SHARES, abs=2e-14)
            assert tuple(float(np.sum(fields.combined * field)) ** 2 for field in domains) == pytest.approx(ENERGY_SHARES, abs=2e-14)
            counts = tuple(int(np.count_nonzero(field)) for field in domains)
            assert counts[0] // 10 == counts[1] // 9 == counts[2] // 6 and counts[0] % 10 == counts[1] % 9 == counts[2] % 6 == 0
            assert len(np.unique(np.round(np.abs(fields.combined[fields.combined != 0.0]), 14))) == 1
            assert not np.any((fields.search != 0.0) & ((fields.fit != 0.0) | (fields.validate != 0.0)))
            assert not np.any((fields.fit != 0.0) & (fields.validate != 0.0))
            height, width = shape
            assert all(np.count_nonzero(fields.search[row * height // 2:(row + 1) * height // 2, column * width // 2:(column + 1) * width // 2]) for row in range(2) for column in range(2))
            for field, tile_ids in ((fields.fit, FIT_TILE_IDS), (fields.validate, VALIDATE_TILE_IDS)):
                assert all(np.count_nonzero(field[(tile_id // 4) * height // 4:(tile_id // 4 + 1) * height // 4, (tile_id % 4) * width // 4:(tile_id % 4 + 1) * width // 4]) for tile_id in tile_ids)

    direct = {"search": b"search-a", "fit": b"fit-a", "validate": b"validate-a"}
    baseline = method._domain_fields((64, 64), direct)
    for changed_name in direct:
        changed = {**direct, changed_name: direct[changed_name] + b"-changed"}
        perturbed = method._domain_fields((64, 64), changed)
        for field_name in direct:
            equal = np.array_equal(getattr(baseline, field_name), getattr(perturbed, field_name))
            assert equal is (field_name != changed_name)


@pytest.mark.unit
def test_keyed_bipolar_microcode_is_per_pixel_balanced_and_has_bounded_sidelobes() -> None:
    mask = np.ones((5, 5), dtype=bool)
    first = method._balanced_bipolar_code(mask, b"domain-a", b"component:cell")
    repeated = method._balanced_bipolar_code(mask, b"domain-a", b"component:cell")
    other_domain = method._balanced_bipolar_code(mask, b"domain-b", b"component:cell")
    assert np.array_equal(first, repeated) and not np.array_equal(first, other_domain)
    assert np.count_nonzero(first > 0.0) == np.count_nonzero(first < 0.0) == 12
    assert np.count_nonzero(first == 0.0) == 1 and float(np.sum(first)) == 0.0
    assert np.linalg.norm(first) == pytest.approx(math.sqrt(24.0))

    search_support, _ = method._domain_support_masks((96, 96))
    component = method._sparse_prn_component((96, 96), b"domain-a", b"search:component", 8, 2, support=search_support)
    assert np.linalg.norm(component) == pytest.approx(1.0, abs=2e-14)
    assert float(np.sum(component)) == pytest.approx(0.0, abs=2e-14)
    correlations = [float(np.sum(component * np.roll(component, shift, axis=(0, 1)))) for shift in ((0, 1), (1, 0), (1, 1), (2, 3))]
    assert max(abs(value) for value in correlations) < .75
    for row in range(2):
        for column in range(2):
            patch = component[row * 48:(row + 1) * 48, column * 48:(column + 1) * 48]
            assert np.count_nonzero(patch) > 0


@pytest.mark.unit
def test_rgb_and_decoder_output_writers_keep_frozen_budget_and_single_hook() -> None:
    ordinary = np.full((64, 64, 3), .5)
    marked, budget = method.write_g1r_rgb(ordinary, KEY)
    assert not np.array_equal(marked, ordinary)
    assert budget["luma_rms"] <= budget["luma_rms_cap"] == 2 / 255
    assert budget["luma_peak"] <= budget["luma_peak_cap"] == 8 / 255
    assert budget["luma_rms"] == pytest.approx(WRITER_TARGET_RMS_FRACTION * 2 / 255)
    assert budget["carrier_rms"] == pytest.approx(WRITER_TARGET_RMS_FRACTION * 2 / 255)
    assert budget["rgb_channel_rms_max"] <= budget["rgb_channel_rms_cap"] == RGB_CHANNEL_RMS_CAP
    assert budget["rgb_channel_peak"] <= budget["rgb_channel_peak_cap"] == RGB_CHANNEL_PEAK_CAP
    decoded = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    updated = method.write_g1r_decoder_output(decoded, KEY)
    final_rgb_delta = updated[0].permute(1, 2, 0).numpy() / 2.0
    final_luma_delta = method._luma(final_rgb_delta)
    final_carrier_delta = method._carrier_plane(final_rgb_delta)
    assert float(np.sqrt(np.mean(final_luma_delta**2))) <= 2 / 255
    assert float(np.max(np.abs(final_luma_delta))) <= 8 / 255
    assert float(np.sqrt(np.mean(final_carrier_delta**2))) == pytest.approx(WRITER_TARGET_RMS_FRACTION * 2 / 255, rel=1e-5)
    assert float(np.max(np.sqrt(np.mean(final_rgb_delta**2, axis=(0, 1))))) <= RGB_CHANNEL_RMS_CAP
    assert float(np.max(np.abs(final_rgb_delta))) <= RGB_CHANNEL_PEAK_CAP

    hook = G1RDecoderOutputHook(KEY)
    assert not torch.equal(hook(None, (), decoded), decoded)
    assert hook.writer_budget is not None and hook.writer_budget["passed"] is True
    with pytest.raises(RuntimeError, match="more than once"):
        hook(None, (), decoded)

    pipeline = _Pipeline()
    pair = run_g1r_sd35_pair(pipeline, "an ordinary test scene", KEY, height=256, width=256, generator=torch.Generator().manual_seed(6201))
    assert pipeline.calls == 2 and pair.clean.tobytes() != pair.marked.tobytes()
    assert pair.writer_budget["measurement"] == "actual_post_cast_pre_PIL_final_RGB_equivalent_float64"
    assert len(pipeline.vae.decoder._forward_hooks) == 0
    assert torch.equal(pipeline.vae.decoder(decoded), decoded)

    failing_pipeline = _Pipeline()
    failing_pipeline.fail_on_call = 2
    with pytest.raises(RuntimeError, match="marked generation failure"):
        run_g1r_sd35_pair(failing_pipeline, "an ordinary test scene", KEY, height=256, width=256, generator=torch.Generator().manual_seed(6201))
    assert len(failing_pipeline.vae.decoder._forward_hooks) == 0


@pytest.mark.unit
def test_decoder_post_cast_update_stays_inside_caps_and_preserves_uniform_domain_shares() -> None:
    assert DECODER_DTYPE_GUARD_EPS_MULTIPLIER == 24.0
    dtypes = (torch.float32, torch.float16, torch.bfloat16)
    for size in (32, 64, 128, 512):
        fields = method.g1r_anchor_fields((size, size), KEY)
        domains = (fields.search, fields.fit, fields.validate)
        for dtype in dtypes:
            decoded = torch.full((1, 3, size, size), .37, dtype=dtype)
            updated, budget = method._write_g1r_decoder_output_with_budget(decoded, KEY)
            actual = (updated.to(torch.float64) - decoded.to(torch.float64)) / 2.0
            rms = torch.sqrt(torch.mean(actual * actual, dim=(0, 2, 3)))
            assert float(torch.max(rms)) <= RGB_CHANNEL_RMS_CAP
            assert float(torch.max(torch.abs(actual))) <= RGB_CHANNEL_PEAK_CAP
            assert torch.equal(actual[:, 0], actual[:, 1]) and torch.equal(actual[:, 1], actual[:, 2])
            post_cast_plane = actual[0, 0].numpy()
            post_cast_projections = np.asarray([float(np.sum(post_cast_plane * field)) for field in domains])
            post_cast_shares = post_cast_projections * post_cast_projections / np.sum(post_cast_projections * post_cast_projections)
            share_tolerance = max(1e-6, float(torch.finfo(dtype).eps))
            assert tuple(post_cast_shares.tolist()) == pytest.approx(ENERGY_SHARES, abs=share_tolerance)
            assert budget["passed"] is True
            assert budget["rgb_channel_rms_max"] <= budget["rgb_channel_rms_cap"] == RGB_CHANNEL_RMS_CAP
            assert budget["rgb_channel_peak"] <= budget["rgb_channel_peak_cap"] == RGB_CHANNEL_PEAK_CAP
            assert tuple(budget["domain_energy_share_targets"].values()) == ENERGY_SHARES
            assert tuple(budget["domain_energy_shares"].values()) == pytest.approx(ENERGY_SHARES, abs=share_tolerance)

            guard = 1.0 - DECODER_DTYPE_GUARD_EPS_MULTIPLIER * float(torch.finfo(dtype).eps)
            guarded = method._g1r_scalar_delta((size, size), KEY) * guard
            projections = np.asarray([float(np.sum(guarded * field)) for field in domains])
            assert tuple((projections * projections / np.sum(projections * projections)).tolist()) == pytest.approx(ENERGY_SHARES, abs=2e-14)


@pytest.mark.unit
def test_final_rgb_observability_uses_all_three_domains_and_frozen_quality_limits() -> None:
    yy, xx = np.mgrid[:64, :64]
    base = .3 + .3 * xx / 63 + .1 * yy / 63
    ordinary = np.stack((base, .9 * base, .8 * base), axis=-1)
    marked, _ = method.write_g1r_rgb(ordinary, KEY)
    observation = method.measure_g1r_final_rgb(ordinary, marked, KEY, b"wrong-key-0123456789", lambda image, key: 0.0)
    assert set(observation.correct_domain_scores) == set(observation.wrong_domain_scores) == {"search", "fit", "validate"}
    assert observation.psnr > 40.0 and observation.ssim > .98
    assert observation.luma_rms <= 2 / 255 and observation.luma_peak <= 8 / 255
    assert observation.post_quantization_rgb_channel_rms_max <= RGB_CHANNEL_RMS_CAP and observation.rgb_channel_peak <= RGB_CHANNEL_PEAK_CAP
    assert observation.content_score_drift == 0.0
    assert observation.passed
    assert all(observation.correct_domain_scores[name] > observation.wrong_domain_scores[name] for name in ("search", "fit", "validate"))
    assert DEVELOPMENT_ARTIFACT_FILES == ("g1r-development-records.json", "g1r-development-summary.json", "g1r-development-manifest.json")
    assert DEVELOPMENT_NOTEBOOK_ID == "geometry_v4_g0_g1_colab_v4_g1r_development_v1"
    quantized_diagnostic = method.G1RFinalRGBObservability(
        50.0, .99, 1.0 / 255.0, 1.0 / 255.0, 1.0, 1.0 / 255.0, 0.0,
        {name: 1.0 for name in ("search", "fit", "validate")},
        {name: 0.0 for name in ("search", "fit", "validate")},
    )
    assert quantized_diagnostic.post_quantization_rgb_channel_rms_max == 1.0
    assert quantized_diagnostic.passed


@pytest.mark.unit
def test_validate_domain_cannot_change_search_candidate_or_frozen_h(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = np.eye(3, dtype=np.float64)
    candidate = {"angle": 0.0, "scale": 1.0, "canonical_to_attacked": identity, "rank": (1,), "ncc": 1.0}
    fit = {"valid": True, "canonical_to_attacked": identity, "support": 8, "rank": (1,), "coverage": 1.0, "macro_regions": 4, "reprojection": 0.0, "condition": 1.0, "inlier_ratio": 1.0, "matches": (), "search": candidate}
    monkeypatch.setattr(method, "_search_candidates", lambda image, key: (candidate,))
    monkeypatch.setattr(method, "_fit_candidate", lambda image, item, key: fit)
    monkeypatch.setattr(method, "_holdout_metrics", lambda image, h, key: {"passed": key == b"accept"})
    image = np.full((64, 64, 3), .5)
    accepted = method._detect_domains(image, {"search": b"s", "fit": b"f", "validate": b"accept"})
    rejected = method._detect_domains(image, {"search": b"s", "fit": b"f", "validate": b"reject"})
    assert accepted["search_identity"] == rejected["search_identity"]
    assert accepted["frozen_h"] == rejected["frozen_h"]
    assert accepted["observation"].status == "RELIABLE"
    assert rejected["observation"].status == "UNRELIABLE" and rejected["observation"].H_hat is None


@pytest.mark.unit
def test_detector_has_no_oracle_surface_and_preserves_original_fit_gates() -> None:
    assert tuple(inspect.signature(method.detect_g1r).parameters) == ("attacked_rgb", "detection_key")
    output = method.detect_g1r(np.full((64, 64, 3), .5), KEY)
    assert set(output) == {"H_hat", "corners_hat", "support", "reliability", "status"}
    _, diagnostics = method._detect_g1r_engineering(np.full((64, 64, 3), .5), KEY)
    assert set(diagnostics) == {"search_top_k", "selected_fit", "holdout"}
    assert len(diagnostics["search_top_k"]) <= 5
    serialized = json.dumps(diagnostics, allow_nan=False, sort_keys=True)
    assert "H_hat" not in diagnostics and "key" not in serialized.lower()
    assert KEY.hex() not in serialized and KEY.decode("ascii") not in serialized
    assert len(diagnostics["selected_fit"]["prethreshold_tiles"]) == 8
    assert sum(diagnostics["selected_fit"]["rejection_counts"].values()) == 8
    assert all(set(item) == {"tile_id", "best_correlation", "margin", "accepted", "rejection"} for item in diagnostics["selected_fit"]["prethreshold_tiles"])
    forbidden = {"truth", "original", "clean", "residual", "latent", "attack"}
    assert not forbidden & set(inspect.signature(method._search_candidates).parameters)
    for blind_stage in (method._translation_surface, method._tile_matches, method._holdout_metrics):
        source = inspect.getsource(blind_stage)
        assert "_carrier_plane" in source
    assert FIT_GATES["support"] == 6 and FIT_GATES["coverage"] == .75 and FIT_GATES["macro_regions"] == 3
    assert FIT_GATES["condition"] == 1e4 and FIT_GATES["reprojection"] == .02
    assert FIT_GATES["correlation"] >= .42 and FIT_GATES["margin"] >= .025
    assert LOCAL_PREPROCESSING == "fixed_cubic_polynomial_detrend_then_narrow_band"
    assert HOLDOUT_GATES["psr"] >= 8.0 and TRANSLATION_PSR_MIN >= 8.0
    assert HOLDOUT_GATES["rotation_spread"] == 2.0 and HOLDOUT_GATES["log_scale_spread"] == .03
    assert not hasattr(method, "_rs_spectral_score")
