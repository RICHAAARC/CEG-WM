from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method import geometry_v4_g1r as method
from cegwm.protocol.geometry_v4_g1r import (
    CONFIG_SHA256,
    DEVELOPMENT_ARTIFACT_FILES,
    DEVELOPMENT_NOTEBOOK_ID,
    ENERGY_SHARES,
    FIT_GATES,
    FIT_TILE_IDS,
    HOLDOUT_GATES,
    LOCAL_PREPROCESSING,
    TRANSLATION_PSR_MIN,
    VALIDATE_TILE_IDS,
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
    assert set(FIT_TILE_IDS) | set(VALIDATE_TILE_IDS) == set(range(16))
    assert not set(FIT_TILE_IDS) & set(VALIDATE_TILE_IDS)
    assert len(require_split(contract, "development")) == len(require_split(contract, "confirmation")) == 20
    with pytest.raises(ValueError, match="split"):
        require_split(contract, "mixed")


@pytest.mark.unit
def test_key_domains_and_anchor_energy_are_separate_and_deterministic() -> None:
    keys = derive_g1r_keys(KEY)
    assert len(set(keys.values())) == 3
    fields = method.g1r_anchor_fields((64, 64), KEY)
    repeated = method.g1r_anchor_fields((64, 64), KEY)
    assert np.array_equal(fields.combined, repeated.combined)
    assert np.linalg.norm(fields.search) == pytest.approx(1.0)
    assert np.linalg.norm(fields.fit) == pytest.approx(1.0)
    assert np.linalg.norm(fields.validate) == pytest.approx(1.0)
    assert np.sum(fields.search * fields.fit) == pytest.approx(0.0, abs=1e-8)
    assert np.sum(fields.search * fields.validate) == pytest.approx(0.0, abs=1e-8)
    assert np.sum(fields.fit * fields.validate) == pytest.approx(0.0, abs=1e-7)

    direct = {"search": b"search-a", "fit": b"fit-a", "validate": b"validate-a"}
    baseline = method._domain_fields((64, 64), direct)
    for changed_name in direct:
        changed = {**direct, changed_name: direct[changed_name] + b"-changed"}
        perturbed = method._domain_fields((64, 64), changed)
        for field_name in direct:
            equal = np.array_equal(getattr(baseline, field_name), getattr(perturbed, field_name))
            assert equal is (field_name != changed_name)


@pytest.mark.unit
def test_rgb_and_decoder_output_writers_keep_frozen_budget_and_single_hook() -> None:
    ordinary = np.full((64, 64, 3), .5)
    marked, budget = method.write_g1r_rgb(ordinary, KEY)
    assert not np.array_equal(marked, ordinary)
    assert budget["luma_rms"] <= budget["luma_rms_cap"] == 2 / 255
    assert budget["luma_peak"] <= budget["luma_peak_cap"] == 8 / 255
    decoded = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    updated = method.write_g1r_decoder_output(decoded, KEY)
    final_luma_delta = updated[0, 0].numpy() / 2.0
    assert float(np.sqrt(np.mean(final_luma_delta**2))) <= 2 / 255
    assert float(np.max(np.abs(final_luma_delta))) <= 8 / 255

    hook = G1RDecoderOutputHook(KEY)
    assert not torch.equal(hook(None, (), decoded), decoded)
    with pytest.raises(RuntimeError, match="more than once"):
        hook(None, (), decoded)

    pipeline = _Pipeline()
    pair = run_g1r_sd35_pair(pipeline, "an ordinary test scene", KEY, height=256, width=256, generator=torch.Generator().manual_seed(6201))
    assert pipeline.calls == 2 and pair.clean.tobytes() != pair.marked.tobytes()
    assert len(pipeline.vae.decoder._forward_hooks) == 0
    assert torch.equal(pipeline.vae.decoder(decoded), decoded)

    failing_pipeline = _Pipeline()
    failing_pipeline.fail_on_call = 2
    with pytest.raises(RuntimeError, match="marked generation failure"):
        run_g1r_sd35_pair(failing_pipeline, "an ordinary test scene", KEY, height=256, width=256, generator=torch.Generator().manual_seed(6201))
    assert len(failing_pipeline.vae.decoder._forward_hooks) == 0


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
    assert observation.content_score_drift == 0.0
    assert observation.passed
    assert all(observation.correct_domain_scores[name] > observation.wrong_domain_scores[name] for name in ("search", "fit", "validate"))
    assert DEVELOPMENT_ARTIFACT_FILES == ("g1r-development-records.json", "g1r-development-summary.json", "g1r-development-manifest.json")
    assert DEVELOPMENT_NOTEBOOK_ID == "geometry_v4_g0_g1_colab_v4_g1r_development_v1"


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
    forbidden = {"truth", "original", "clean", "residual", "latent", "attack"}
    assert not forbidden & set(inspect.signature(method._search_candidates).parameters)
    assert FIT_GATES["support"] == 6 and FIT_GATES["coverage"] == .75 and FIT_GATES["macro_regions"] == 3
    assert FIT_GATES["condition"] == 1e4 and FIT_GATES["reprojection"] == .02
    assert FIT_GATES["correlation"] >= .42 and FIT_GATES["margin"] >= .025
    assert LOCAL_PREPROCESSING == "fixed_cubic_polynomial_detrend_then_narrow_band"
    assert HOLDOUT_GATES["psr"] >= 8.0 and TRANSLATION_PSR_MIN >= 8.0
    assert HOLDOUT_GATES["rotation_spread"] == 2.0 and HOLDOUT_GATES["log_scale_spread"] == .03
