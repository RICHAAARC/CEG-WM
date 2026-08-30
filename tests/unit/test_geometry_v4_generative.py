from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from PIL import Image

from cegwm.method import geometry_v4_generative as generated
from cegwm.method.geometry_v4_generative import FrozenWeightedJointContentAdapter, detect_g1_geometry, measure_final_rgb, rectify_g1_rgb, rgb_anchor_basis, rgb_only_anchor_score, write_final_latent_anchor
from cegwm.protocol.geometry_v4_generative import CALLBACK_STEP_INDEX, LUMA_PEAK_CAP, LUMA_RMS_CAP, load_g0_g1_contract

ROOT = Path(__file__).resolve().parents[2]
KEY, WRONG = "0123456789abcdef", "fedcba9876543210"

class _VAE(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.ones((), dtype=torch.float16)); self.seen = None
    def decode(self, value, return_dict=True):
        self.seen = value; return SimpleNamespace(sample=value[:, :3] * self.weight)
class _Pipeline: vae = _VAE()


@pytest.mark.unit
def test_contract_freezes_sole_placement_budget_and_rosters() -> None:
    contract = load_g0_g1_contract(ROOT)
    assert contract["identity"]["callback_step_index_zero_based"] == CALLBACK_STEP_INDEX == 19
    assert tuple(contract["g0"]["seeds"]) == (5101, 5102, 5103, 5104)
    assert tuple(contract["g1"]["seeds"]) == (6101, 6102, 6103, 6104)
    assert tuple(contract["g1"]["attacks"]) == ("identity", "rotation_5", "scale_0.9", "translation_0.08_0", "crop_0.9")
    assert contract["g1_detector"]["h_direction"] == "attacked_to_canonical"
    assert contract["g1_detector"]["min_anchor_score"] == 3.0
    assert tuple(contract["residual_budget"]["global_local_energy_shares"]) == (.4, .6)
    assert LUMA_RMS_CAP == 2 / 255 and LUMA_PEAK_CAP == 8 / 255


@pytest.mark.unit
def test_writer_is_keyed_deterministic_and_rejects_non_latent_input() -> None:
    latents = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
    first = write_final_latent_anchor(latents, KEY, _Pipeline())
    assert torch.equal(first, write_final_latent_anchor(latents, KEY, _Pipeline()))
    assert not torch.equal(first, write_final_latent_anchor(latents, WRONG, _Pipeline()))
    with pytest.raises(ValueError, match="NCHW"):
        write_final_latent_anchor(torch.zeros((4, 8, 8)), KEY, _Pipeline())

@pytest.mark.unit
def test_shared_basis_is_keyed_and_signed() -> None:
    basis, global_part, local_part = rgb_anchor_basis((32, 32), KEY)
    wrong, _, _ = rgb_anchor_basis((32, 32), WRONG)
    assert torch.equal(basis, rgb_anchor_basis((32, 32), KEY)[0]) and not torch.equal(basis, wrong)
    assert abs(float((global_part * local_part).sum())) < 1e-5
    image = np.full((32, 32, 3), .5); image += basis[0, 0].numpy()[..., None] * .001
    assert rgb_only_anchor_score(image, KEY) > rgb_only_anchor_score(image, WRONG)

@pytest.mark.unit
def test_vae_parameter_dtype_cast_remains_differentiable() -> None:
    pipeline = _Pipeline(); latents = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
    updated = write_final_latent_anchor(latents, KEY, pipeline)
    assert pipeline.vae.seen.dtype == torch.float16
    assert torch.linalg.vector_norm(updated - latents) > 0


@pytest.mark.unit
def test_rgb_observability_is_rgb_key_only_and_fail_closed_on_equal_keys() -> None:
    clean = np.full((64, 64, 3), .5, dtype=np.float64)
    marked = clean.copy()
    yy, xx = np.mgrid[:64, :64]
    marked += (0.0001 * np.cos(2 * np.pi * 8 * xx / 64))[..., None]
    observation = measure_final_rgb(clean, marked, KEY, WRONG, lambda rgb, key: float(rgb.mean()))
    assert observation.luma_rms <= LUMA_RMS_CAP and observation.luma_peak <= LUMA_PEAK_CAP
    assert np.isfinite(rgb_only_anchor_score(marked, KEY))
    with pytest.raises(ValueError, match="must differ"):
        measure_final_rgb(clean, marked, KEY, KEY, lambda rgb, key: 0.0)


@pytest.mark.unit
def test_reused_content_adapter_uses_lf_hf_and_weighted_joint_current_rgb_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object, bytes]] = []
    monkeypatch.setattr(generated, "score_content_whitened_lf_image", lambda image, key, assets: calls.append(("lf", image, key)) or .2)
    monkeypatch.setattr(generated, "score_hf_image", lambda image, key, assets: calls.append(("hf", image, key)) or .3)
    monkeypatch.setattr(generated, "weighted_joint_score", lambda lf, hf, asset: .25)
    adapter = FrozenWeightedJointContentAdapter(object(), object(), object(), "asset.json", "a" * 64)  # type: ignore[arg-type]
    assert adapter(np.full((16, 16, 3), .5), b"normalized") == .25
    assert [call[0] for call in calls] == ["lf", "hf"] and all(call[2] == b"normalized" for call in calls)
    with pytest.raises(ValueError, match="finite current RGB"):
        adapter(np.full((16, 16, 3), np.nan), b"normalized")
    with pytest.raises(TypeError, match="normalized detection-key bytes"):
        adapter(np.zeros((16, 16, 3)), "not-bytes")  # type: ignore[arg-type]


@pytest.mark.unit
def test_g1_blind_detector_recovers_five_fixed_attack_classes_and_wrong_key_fails_closed() -> None:
    from experiments.geometry_v4_generative_engine import _attack

    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    base = .45 + .04 * np.sin(2 * np.pi * xx / width) + .03 * np.cos(2 * np.pi * yy / height)
    basis = rgb_anchor_basis((height, width), KEY)[0][0, 0].numpy()
    marked = np.repeat((base + 5.0 * basis)[..., None], 3, axis=2).clip(0.0, 1.0)
    marked_image = Image.fromarray((marked * 255).round().astype(np.uint8), mode="RGB")
    for attack_name in ("identity", "rotation_5", "scale_0.9", "translation_0.08_0", "crop_0.9"):
        attacked = np.asarray(_attack(marked_image, attack_name), dtype=np.float64) / 255.0
        correct = detect_g1_geometry(attacked, KEY)
        wrong = detect_g1_geometry(attacked, WRONG)
        assert correct["status"] == "RELIABLE", (attack_name, correct["diagnostics"])
        assert correct["H_hat"] is not None and len(correct["corners_hat"]) == 4 and correct["support"] >= 6
        assert wrong["status"] == "UNRELIABLE", (attack_name, wrong["diagnostics"])
        rectified = rectify_g1_rgb(attacked, correct["H_hat"])
        assert rgb_only_anchor_score(rectified, KEY) > rgb_only_anchor_score(rectified, WRONG)
        if attack_name != "identity":
            assert np.mean(np.square(rectified - marked)) < np.mean(np.square(attacked - marked)), attack_name


@pytest.mark.unit
def test_g1_detector_public_interface_has_no_oracle_or_attack_input() -> None:
    import inspect

    assert tuple(inspect.signature(detect_g1_geometry).parameters) == ("attacked_rgb", "detection_key")
    unreliable = detect_g1_geometry(np.full((64, 64, 3), .5), KEY)
    assert unreliable["status"] == "UNRELIABLE" and unreliable["H_hat"] is None and unreliable["corners_hat"] == ()
