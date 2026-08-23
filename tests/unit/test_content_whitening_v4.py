from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

import pytest
import torch

from cegwm.method import content_whitening_v4 as v4

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / v4.FIT_MANIFEST_REPO_PATH
_PRODUCER_EXACT = "1234567890abcdef1234567890abcdef12345678"


def _observations() -> tuple[torch.Tensor, ...]:
    axis = torch.arange(64, dtype=torch.float32)
    y = axis[:, None]
    x = axis[None, :]
    values = []
    for unit in range(v4.FIT_UNIT_COUNT):
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


@pytest.fixture(scope="module")
def fitted() -> tuple[v4.FitManifest, v4.FitProtocolBinding, v4.WhiteningFit, v4.WhiteningAsset]:
    manifest = v4.load_fit_manifest(_MANIFEST)
    binding = v4.bind_fit_protocol(manifest, _PRODUCER_EXACT)
    fit = v4.fit_whitening_operator(_observations())
    asset = v4.build_whitening_asset(binding, fit.words_be_hex)
    return manifest, binding, fit, asset


@pytest.mark.unit
def test_manifest_archive_sha_order_and_v3_disjointness() -> None:
    manifest = v4.load_fit_manifest(_MANIFEST)
    assert hashlib.sha256(_MANIFEST.read_bytes()).hexdigest() == manifest.raw_sha256
    assert v4.ARCHIVE_MANIFEST_SHA256 == (
        "5d7388a92c98aa5fb1996369bae8de65360e2d25fa7569400135753257bb6e86"
    )
    assert len(manifest.entries) == 32
    assert manifest.entries[0].cluster_identity == "alabaster_alabaster"
    assert manifest.entries[-1].cluster_identity == "frosted_frosted"
    assert tuple(item.cluster_ordinal for item in manifest.entries) == tuple(range(32))
    assert tuple(item.generation_seed for item in manifest.entries) == tuple(
        range(2026081000, 2026081032)
    )
    fit_tuples = {item.generation_tuple for item in manifest.entries}
    assert fit_tuples.isdisjoint(v4.V3_FORMAL_DENY_TUPLES)
    assert v4.V3_CANARY_DENY_TUPLE not in fit_tuples
    assert len(fit_tuples) == 32


@pytest.mark.unit
@pytest.mark.parametrize("deny_name", ["V3_FORMAL_DENY_TUPLES", "V3_CANARY_DENY_TUPLE"])
def test_manifest_explicitly_rejects_v3_overlap(
    monkeypatch: pytest.MonkeyPatch,
    deny_name: str,
) -> None:
    entry_tuple = v4.load_fit_manifest(_MANIFEST).entries[0].generation_tuple
    if deny_name == "V3_FORMAL_DENY_TUPLES":
        monkeypatch.setattr(v4, deny_name, (entry_tuple,))
        match = "V3 formal roster"
    else:
        monkeypatch.setattr(v4, deny_name, entry_tuple)
        match = "V3 canary"
    with pytest.raises(ValueError, match=match):
        v4.load_fit_manifest(_MANIFEST)


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["missing", "extra", "payload_drift"])
def test_manifest_missing_extra_and_payload_drift_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    value = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    if mutation == "missing":
        del value["entries"][0]["prompt_digest"]
        match = "fields differ"
    elif mutation == "extra":
        value["entries"][0]["unexpected"] = "forbidden"
        match = "fields differ"
    else:
        value["entries"][0]["prompt"] += " drift"
        match = "prompt digest differs"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        v4.load_fit_manifest(path)


@pytest.mark.unit
def test_orthonormal_dct_chebyshev_rings_hex_and_digest(
    fitted: tuple[v4.FitManifest, v4.FitProtocolBinding, v4.WhiteningFit, v4.WhiteningAsset],
) -> None:
    manifest, binding, fit, asset = fitted
    dct = v4._dct_matrix()
    assert torch.allclose(dct @ dct.T, torch.eye(64, dtype=torch.float64), atol=1e-12, rtol=0.0)
    masks = v4._ring_masks()
    assert tuple(int(mask.sum()) for mask in masks) == v4.CHEBYSHEV_RING_COUNTS
    assert sum(v4.CHEBYSHEV_RING_COUNTS) == 4095
    assert not any(bool(mask[0, 0]) for mask in masks)
    assert math.isfinite(fit.energy_global) and fit.energy_global > 0.0
    assert fit.ridge == math.ldexp(fit.energy_global, -10)
    assert len(fit.words_be_hex) == 96
    assert all(re.fullmatch(r"[0-9a-f]{8}", word) for word in fit.words_be_hex)
    decoded = tuple(struct.unpack(">f", bytes.fromhex(word))[0] for word in fit.words_be_hex)
    assert all(math.isfinite(value) and value > 0.0 for value in decoded)
    assert binding.run_id.startswith("content-v4-whitening-fit-")
    assert binding.producer_exact == _PRODUCER_EXACT
    assert hashlib.sha256(v4.stable_json_bytes(binding.payload)).hexdigest() == binding.digest
    assert asset.json_bytes == v4.stable_json_bytes(asset.payload)
    assert hashlib.sha256(asset.json_bytes).hexdigest() == asset.digest
    assert asset.payload["fit_contract"]["fit_manifest"]["raw_sha256"] == manifest.raw_sha256
    assert asset.payload["fit_contract"]["scientific_denominator"] == 0
    assert asset.payload["whitening_words_be_hex_channel_major_band_minor"] == list(
        fit.words_be_hex
    )
    assert "energy_global" not in asset.payload
    assert "ridge" not in asset.payload


@pytest.mark.unit
def test_protocol_is_pre_w_and_asset_digest_depends_on_final_words(
    fitted: tuple[v4.FitManifest, v4.FitProtocolBinding, v4.WhiteningFit, v4.WhiteningAsset],
) -> None:
    manifest, binding, fit, asset = fitted
    rebound = v4.bind_fit_protocol(manifest, _PRODUCER_EXACT)
    assert rebound.digest == binding.digest and rebound.run_id == binding.run_id
    assert "whitening_words_be_hex_channel_major_band_minor" not in binding.payload
    other_exact = "2234567890abcdef1234567890abcdef12345678"
    changed_binding = v4.bind_fit_protocol(manifest, other_exact)
    assert changed_binding.digest != binding.digest
    assert changed_binding.run_id != binding.run_id
    changed_words = list(fit.words_be_hex)
    changed_words[0] = struct.pack(">f", struct.unpack(">f", bytes.fromhex(changed_words[0]))[0] * 2.0).hex()
    changed = v4.build_whitening_asset(binding, changed_words)
    assert changed.digest != asset.digest


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
    binding = v4.bind_fit_protocol(v4.load_fit_manifest(_MANIFEST), _PRODUCER_EXACT)
    with pytest.raises(ValueError, match=match):
        v4.build_whitening_asset(binding, words)
