from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import struct
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from cegwm.method import content_whitening_v4 as v4
from cegwm.method.lf import (
    LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
    LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
    FrozenLFPublicAssets,
    reconstruct_lf_carrier,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / v4.FIT_MANIFEST_REPO_PATH
_PRODUCER_EXACT = "1234567890abcdef1234567890abcdef12345678"
_ORACLE_WORD_BYTES_SHA256 = (
    "1f58fed97a0e6899e7ea164f0e70466cdb62b47abc5e99cff74bb7c96f8d8158"
)
_ORACLE_ORDER_SENTINELS = {
    0: "3f5357b7",
    1: "3ddfb58e",
    5: "419215dc",
    6: "3edf1687",
    17: "419055a8",
    47: "4191600e",
    48: "3f134fc8",
    77: "41917d5c",
    90: "3f535432",
    95: "4191bc34",
}


def _observations() -> tuple[torch.Tensor, ...]:
    axis = torch.arange(64, dtype=torch.float32)
    y = axis[:, None]
    x = axis[None, :]
    values = []
    for unit in range(32):
        channels = []
        for channel in range(16):
            field = (
                torch.sin((channel % 5 + 1) * math.pi * (x + 0.5) / 64.0)
                + torch.cos((unit % 7 + 1) * math.pi * (y + 0.5) / 64.0)
                + 0.01 * ((x * y + unit * x + channel * y) % 17.0)
            )
            channels.append(field)
        values.append(torch.stack(channels).unsqueeze(0).contiguous())
    return tuple(values)


def _oracle_affine_residual(values: np.ndarray) -> np.ndarray:
    """Independent normalized-coordinate affine LS, batched only over RHS columns."""

    if values.shape[-2:] != (64, 64):
        raise ValueError("oracle expects 64x64 fields")
    coordinate = (2.0 * np.arange(64, dtype=np.float64) - 63.0) / 63.0
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    design = np.stack((np.ones((64, 64)), y, x), axis=-1).reshape(4096, 3)
    flattened = values.astype(np.float64, copy=False).reshape(-1, 4096).T
    coefficients, _, rank, _ = np.linalg.lstsq(design, flattened, rcond=None)
    if rank != 3:
        raise AssertionError("oracle affine design lost full rank")
    residual = flattened - design @ coefficients
    return residual.T.reshape(values.shape)


def _oracle_dct_matrix() -> np.ndarray:
    n = np.arange(64, dtype=np.float64)
    k = np.arange(64, dtype=np.float64)[:, None]
    alpha = np.full((64, 1), np.sqrt(2.0 / 64.0), dtype=np.float64)
    alpha[0, 0] = np.sqrt(1.0 / 64.0)
    return alpha * np.cos((np.pi / 64.0) * (n + 0.5) * k)


def _oracle_ring_masks() -> tuple[np.ndarray, ...]:
    axis = np.arange(64)
    radius = np.maximum(axis[:, None], axis[None, :])
    return tuple(
        (radius >= lower) & (radius < upper)
        for lower, upper in ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
    )


def _oracle_whitening_words(
    observations: tuple[torch.Tensor, ...],
) -> tuple[tuple[str, ...], np.ndarray, float, float]:
    """Independent NumPy oracle for every fitted scalar and serialized word."""

    values = np.stack([item.numpy()[0] for item in observations], axis=0).astype(
        np.float64, copy=False
    )
    residual = _oracle_affine_residual(values)
    dct = _oracle_dct_matrix()
    coefficients = np.matmul(np.matmul(dct, residual), dct.T)
    masks = _oracle_ring_masks()
    counts = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    energy = np.stack(
        [
            np.square(coefficients[..., mask]).sum(axis=(0, 2))
            / (32 * int(count))
            for mask, count in zip(masks, counts, strict=True)
        ],
        axis=1,
    )
    energy_global = float(np.sum(energy * counts) / (16 * 4095))
    ridge = float(np.ldexp(energy_global, -10))
    weights = np.asarray(np.power(energy + ridge, -0.5), dtype=np.float32)
    encoded = np.asarray(weights, dtype=">f4").tobytes(order="C").hex()
    words = tuple(encoded[index : index + 8] for index in range(0, len(encoded), 8))
    return words, energy, energy_global, ridge


@pytest.fixture(scope="module")
def fitted() -> tuple[v4.FitManifest, v4.WhiteningFit, v4.WhiteningAsset]:
    manifest = v4.load_fit_manifest(_MANIFEST)
    fit = v4.fit_whitening_operator(_observations())
    asset = v4.build_whitening_asset(_PRODUCER_EXACT, fit.words_be_hex)
    return manifest, fit, asset


@pytest.mark.unit
def test_independent_oracle_removes_closed_form_planes_and_preserves_residual() -> None:
    coordinate = (2.0 * np.arange(64, dtype=np.float64) - 63.0) / 63.0
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    intercept = np.array([[0.5, -1.25, 2.0], [3.0, -0.75, 1.5]])[..., None, None]
    slope_y = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])[..., None, None]
    slope_x = np.array([[-0.7, 0.8, -0.9], [1.0, -1.1, 1.2]])[..., None, None]
    planes = intercept + slope_y * y + slope_x * x
    removed = _oracle_affine_residual(planes)
    np.testing.assert_allclose(removed, np.zeros_like(removed), atol=2e-14, rtol=0.0)

    dct = _oracle_dct_matrix()
    residual_basis = np.outer(dct[2], dct[4])
    scale = np.array([[0.25, -0.5, 0.75], [1.0, -1.25, 1.5]])[..., None, None]
    expected_residual = scale * residual_basis
    recovered = _oracle_affine_residual(planes + expected_residual)
    np.testing.assert_allclose(recovered, expected_residual, atol=2e-14, rtol=0.0)


@pytest.mark.unit
def test_complete_production_words_match_independent_oracle_and_fixed_golden(
    fitted: tuple[v4.FitManifest, v4.WhiteningFit, v4.WhiteningAsset],
) -> None:
    _, production, _ = fitted
    oracle_words, oracle_energy, oracle_global, oracle_ridge = (
        _oracle_whitening_words(_observations())
    )
    assert oracle_energy.shape == (16, 6)
    assert np.isfinite(oracle_energy).all() and bool((oracle_energy >= 0.0).all())
    assert oracle_global > 0.0
    assert oracle_ridge == np.ldexp(oracle_global, -10)
    assert len(oracle_words) == 96
    assert all(re.fullmatch(r"[0-9a-f]{8}", word) for word in oracle_words)
    expected_float32 = np.asarray(
        np.power(oracle_energy + oracle_ridge, -0.5), dtype=np.float32
    )
    decoded_big_endian = np.frombuffer(
        bytes.fromhex("".join(oracle_words)), dtype=">f4"
    ).astype(np.float32).reshape(16, 6)
    np.testing.assert_array_equal(decoded_big_endian, expected_float32)
    assert production.words_be_hex == oracle_words
    golden_digest = hashlib.sha256(bytes.fromhex("".join(oracle_words))).hexdigest()
    assert golden_digest == _ORACLE_WORD_BYTES_SHA256
    assert {
        index: oracle_words[index] for index in _ORACLE_ORDER_SENTINELS
    } == _ORACLE_ORDER_SENTINELS


@pytest.mark.unit
def test_manifest_has_exact_ordered_unique_fit_bindings() -> None:
    manifest = v4.load_fit_manifest(_MANIFEST)
    assert len(manifest.entries) == 32
    assert manifest.entries[0].unit_id == "alabaster_alabaster"
    assert manifest.entries[-1].unit_id == "frosted_frosted"
    assert tuple(item.generation_seed for item in manifest.entries) == tuple(
        range(2026081000, 2026081032)
    )
    assert len({item.unit_id for item in manifest.entries}) == 32
    assert len({(item.prompt, item.generation_seed) for item in manifest.entries}) == 32


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["count", "duplicate"])
def test_manifest_count_and_uniqueness_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    value = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    if mutation == "count":
        value["entries"].pop()
        match = "exactly 32"
    else:
        value["entries"][1]["unit_id"] = value["entries"][0]["unit_id"]
        match = "must be unique"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        v4.load_fit_manifest(path)


@pytest.mark.unit
def test_compact_asset_round_trip(
    tmp_path: Path,
    fitted: tuple[v4.FitManifest, v4.WhiteningFit, v4.WhiteningAsset],
) -> None:
    _, fit, asset = fitted
    assert set(asset.payload) == {
        "schema_version",
        "observation_contract_id",
        "whitening_shape",
        "whitening_order",
        "whitening_words_be_hex",
        "fit_sample_count",
        "producer_exact",
    }
    assert asset.payload["whitening_shape"] == [16, 6]
    assert asset.payload["whitening_order"] == "channel_major_band_minor"
    assert asset.payload["whitening_words_be_hex"] == list(fit.words_be_hex)
    path = tmp_path / "asset.json"
    path.write_bytes(asset.json_bytes)
    assert v4.load_whitening_asset(path) == asset


@pytest.mark.unit
@pytest.mark.parametrize("defect", ["count", "shape", "dtype", "order", "nonfinite"])
def test_observation_contract_defects_fail_closed(defect: str) -> None:
    values = list(_observations())
    match = "exactly 32"
    if defect == "count":
        values.pop()
    elif defect == "shape":
        values[0] = values[0][..., :63]
        match = "shape differs"
    elif defect == "dtype":
        values[0] = values[0].to(torch.float64)
        match = "CPU float32"
    elif defect == "order":
        values[0] = values[0].transpose(-1, -2)
        match = "C order"
    else:
        values[0][0, 0, 0, 0] = float("nan")
        match = "finite"
    with pytest.raises((TypeError, ValueError), match=match):
        v4.fit_whitening_operator(values)


@pytest.mark.unit
def test_degenerate_affine_observations_fail_closed() -> None:
    observation = torch.ones(v4.OBSERVATION_SHAPE, dtype=torch.float32)
    with pytest.raises(ValueError, match="global whitening energy"):
        v4.fit_whitening_operator(tuple(observation.clone() for _ in range(32)))


@pytest.mark.unit
@pytest.mark.parametrize(
    "words,match",
    [
        (("3f800000",) * 95, "exactly 96"),
        (("3F800000",) * 96, "lowercase 8-hex"),
        (("7f800000",) * 96, "finite positive"),
    ],
)
def test_whitening_word_contract_fails_closed(words: tuple[str, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        v4.build_whitening_asset(_PRODUCER_EXACT, words)


class _DetectorProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        self.calls += 1
        pixels = np.asarray(image.resize((64, 64)), dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()


class _DetectorVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=0.5, shift_factor=0.1)
        self.calls = 0

    def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        mode = torch.cat([pixels] * 5 + [pixels.mean(dim=1, keepdim=True)], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mode))


def _detector_fixture() -> tuple[
    Image.Image,
    bytes,
    v4.FrozenContentV4LFPublicAssets,
    _DetectorProcessor,
    _DetectorVAE,
]:
    y, x = np.mgrid[:64, :64]
    pixels = np.stack(
        ((3 * x + 5 * y) % 256, (7 * x + 11 * y + 13) % 256, (x * x + 3 * y) % 256),
        axis=-1,
    ).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    processor = _DetectorProcessor()
    vae = _DetectorVAE()
    carrier_assets = FrozenLFPublicAssets(
        vae,
        processor,
        "stabilityai/stable-diffusion-3.5-medium:image_processor",
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    asset = v4.load_frozen_content_v4_whitening_asset(_REPO_ROOT)
    return (
        image,
        b"content-v4-detector-golden-key",
        v4.FrozenContentV4LFPublicAssets(carrier_assets, asset),
        processor,
        vae,
    )


def _independent_detector_oracle(
    image: Image.Image,
    key: bytes,
    assets: v4.FrozenContentV4LFPublicAssets,
) -> float:
    pixels = np.asarray(image.resize((64, 64)), dtype=np.float32) / 255.0
    chw = np.transpose(pixels, (2, 0, 1))[None]
    mode = np.concatenate([chw] * 5 + [chw.mean(axis=1, keepdims=True)], axis=1)
    observation = np.asarray((mode - 0.1) * 0.5, dtype=np.float32)
    carrier = reconstruct_lf_carrier(
        key,
        (1, 16, 64, 64),
        assets.carrier_assets,
        dtype=torch.float32,
        device="cpu",
    ).numpy()
    dct = _oracle_dct_matrix()
    observation_dct = np.matmul(
        np.matmul(dct, _oracle_affine_residual(observation)), dct.T
    )
    carrier_dct = np.matmul(np.matmul(dct, _oracle_affine_residual(carrier)), dct.T)
    words = assets.whitening_asset.payload["whitening_words_be_hex"]
    weights = np.frombuffer(bytes.fromhex("".join(words)), dtype=">f4").astype(
        np.float64
    ).reshape(16, 6)
    observation_parts = []
    carrier_parts = []
    for channel in range(16):
        for band, mask in enumerate(_oracle_ring_masks()):
            observation_parts.append(observation_dct[0, channel][mask] * weights[channel, band])
            carrier_parts.append(carrier_dct[0, channel][mask] * weights[channel, band])
    observation_vector = np.concatenate(observation_parts)
    carrier_vector = np.concatenate(carrier_parts)
    return float(
        np.dot(observation_vector, carrier_vector)
        / (np.linalg.norm(observation_vector) * np.linalg.norm(carrier_vector))
    )


@pytest.mark.unit
def test_frozen_repository_asset_pair_exact_and_key_independent(tmp_path: Path) -> None:
    asset = v4.load_frozen_content_v4_whitening_asset(_REPO_ROOT)
    assert hashlib.sha256(asset.json_bytes).hexdigest() == v4.ASSET_SHA256
    assert hashlib.sha256(
        (_REPO_ROOT / v4.ASSET_SIDECAR_REPO_PATH).read_bytes()
    ).hexdigest() == v4.ASSET_SIDECAR_SHA256
    weights = v4.decode_whitening_weights(asset)
    assert tuple(weights.shape) == (16, 6)
    assert weights.dtype == torch.float32
    assert bool(torch.isfinite(weights).all()) and bool((weights > 0.0).all())
    assert set(asset.payload) == {
        "schema_version", "observation_contract_id", "whitening_shape",
        "whitening_order", "whitening_words_be_hex", "fit_sample_count",
        "producer_exact",
    }
    assert asset.payload["schema_version"] == v4.ASSET_SCHEMA_ID
    assert asset.payload["producer_exact"] == v4.ASSET_PRODUCER_EXACT
    assert tuple(field.name for field in fields(v4.FrozenContentV4LFPublicAssets)) == (
        "carrier_assets", "whitening_asset"
    )
    assert tuple(inspect.signature(v4.decode_whitening_weights).parameters) == ("asset",)

    root = tmp_path / "repo"
    destination = root / v4.ASSET_REPO_PATH
    destination.parent.mkdir(parents=True)
    destination.write_bytes(asset.json_bytes)
    sidecar = root / v4.ASSET_SIDECAR_REPO_PATH
    sidecar.write_bytes((_REPO_ROOT / v4.ASSET_SIDECAR_REPO_PATH).read_bytes() + b"x")
    with pytest.raises(ValueError, match="sidecar SHA"):
        v4.load_frozen_content_v4_whitening_asset(root)


@pytest.mark.unit
def test_content_v4_detector_matches_independent_oracle_golden_and_real_boundary() -> None:
    image, key, assets, processor, vae = _detector_fixture()
    oracle = _independent_detector_oracle(image, key, assets)
    assert hashlib.sha256(struct.pack(">d", oracle)).hexdigest() == (
        "ca91ddfa70e2f675f8d5aa994c1c05bb13528e2c5ffa9281131b8a597a0e8453"
    )
    score = v4.score_content_v4_lf_image(image, key, assets)
    assert score == pytest.approx(oracle, rel=0.0, abs=2e-17)
    assert score == pytest.approx(-0.008900470647137703, rel=0.0, abs=1e-18)
    assert processor.calls == 1 and vae.calls == 1
    assert v4.decode_whitening_weights(assets.whitening_asset).equal(
        v4.decode_whitening_weights(assets.whitening_asset)
    )


@pytest.mark.unit
def test_content_v4_detector_is_blind_and_fails_closed_on_degenerate_observation() -> None:
    image, key, assets, _, _ = _detector_fixture()
    assert tuple(inspect.signature(v4.score_content_v4_lf_image).parameters) == (
        "image", "detection_key", "frozen_public_assets"
    )
    with pytest.raises(TypeError):
        v4.score_content_v4_lf_image(image, key, assets, object())

    class _ZeroVAE(_DetectorVAE):
        def encode(self, pixels: torch.Tensor) -> SimpleNamespace:
            mode = torch.zeros((1, 16, 64, 64), dtype=pixels.dtype, device=pixels.device)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: mode))

    zero_vae = _ZeroVAE()
    zero_assets = FrozenLFPublicAssets(
        zero_vae,
        assets.carrier_assets.image_processor,
        assets.carrier_assets.image_processor_id,
        LF_BALANCED_BLOCKS_CARRIER_METHOD_ID,
        LF_BLOCKNORM_DETECTOR_STATISTIC_ID,
        LF_BALANCED_BLOCKS_EVALUATED_CANDIDATE_ID,
    )
    with pytest.raises(ValueError, match="denominator"):
        v4.score_content_v4_lf_image(
            image,
            key,
            v4.FrozenContentV4LFPublicAssets(zero_assets, assets.whitening_asset),
        )
