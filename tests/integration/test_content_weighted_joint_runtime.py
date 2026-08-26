from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from cegwm.protocol.content_calibration import CONTENT_CALIBRATION_CALIBRATION_SPLIT, ContentCalibrationUnit
from cegwm.protocol.content_chain import ContentChainUnit
from cegwm.method.content_weighted_joint import load_calibration_asset
from cegwm.runtime import content_weighted_joint_sd35 as runtime
from cegwm.runtime.content_iss_sd35 import ContentISSRunOutput


def _assets() -> runtime.ContentCalibrationAssets:
    instance = object.__new__(runtime.ContentCalibrationAssets)
    iss = SimpleNamespace(lf_public_assets=object(), hf_public_assets=object())
    object.__setattr__(instance, "iss_assets", iss)
    return instance


@pytest.mark.integration
def test_real_iss_pair_delegation_excludes_candidate_registered_and_orders_33_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Image.new("RGB", (512, 512), "gray")
    primary_null = Image.new("RGB", (512, 512), "white")
    calls: list[tuple[str, object]] = []
    wrong_keys = tuple(bytes([index + 1]) * 32 for index in range(16))
    monkeypatch.setattr(runtime, "derive_calibration_wrong_keys", lambda key: wrong_keys)
    monkeypatch.setattr(
        runtime,
        "run_content_iss_evaluation_pair",
        lambda *args, **kwargs: calls.append(("v6_pair", kwargs))
        or ContentISSRunOutput(candidate, primary_null, object()),
    )
    monkeypatch.setattr(runtime, "require_ordinary_rgb_image", lambda image: image)
    monkeypatch.setattr(
        runtime,
        "score_content_whitened_lf_image",
        lambda image, key, assets: calls.append(("lf", (image, key))) or key[0] / 255,
    )
    monkeypatch.setattr(
        runtime,
        "score_hf_image",
        lambda image, key, assets: calls.append(("hf", (image, key))) or -key[0] / 255,
    )
    unit = ContentCalibrationUnit(
        "content-v9-calibration-0001", CONTENT_CALIBRATION_CALIBRATION_SPLIT,
        "content-v9-calibration-source-0001", "prompt", 2026091000, 512, 512,
    )
    registered = b"r" * 32
    pairs = runtime.run_content_calibration_unit(object(), unit, registered, _assets())
    assert len(pairs) == 33
    assert [name for name, _ in calls].count("v6_pair") == 1
    scorer_calls = [payload for name, payload in calls if name == "lf"]
    assert len(scorer_calls) == 33
    assert all(key != registered for image, key in scorer_calls if image is candidate)
    assert [key for image, key in scorer_calls if image is candidate] == list(wrong_keys)
    assert [key for image, key in scorer_calls if image is primary_null] == [registered, *wrong_keys]


@pytest.mark.integration
def test_runtime_rejects_non_calibration_unit_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime, "run_content_iss_evaluation_pair", lambda *a, **k: pytest.fail("must not run")
    )
    unit = ContentCalibrationUnit("u", "wrong", "s", "p", 1, 512, 512)
    with pytest.raises(TypeError, match="validated calibration unit"):
        runtime.run_content_calibration_unit(object(), unit, b"k" * 32, _assets())


@pytest.mark.integration
def test_stability_runtime_uses_one_real_iss_pair_and_same_asset_for_candidate_and_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    asset_path = root / "configs/content_chain/assets/content_v9_calibrated_weighted_joint_v1.json"
    asset = load_calibration_asset(asset_path, asset_path.with_name(f"{asset_path.name}.sha256"))
    candidate = Image.new("RGB", (512, 512), "gray")
    primary_null = Image.new("RGB", (512, 512), "white")
    measurement = object()
    pair_calls: list[dict[str, object]] = []
    score_calls: list[tuple[Image.Image, bytes]] = []
    monkeypatch.setattr(
        runtime,
        "run_content_iss_evaluation_pair",
        lambda *args, **kwargs: pair_calls.append(kwargs)
        or ContentISSRunOutput(candidate, primary_null, measurement),
    )
    monkeypatch.setattr(runtime, "require_ordinary_rgb_image", lambda image: image)
    monkeypatch.setattr(
        runtime,
        "score_content_whitened_lf_image",
        lambda image, key, assets: score_calls.append((image, key)) or key[0] / 255.0,
    )
    monkeypatch.setattr(
        runtime,
        "score_hf_image",
        lambda image, key, assets: -(key[0] / 255.0),
    )
    unit = ContentChainUnit("u", "s", "source", "prompt", 7, 512, 512)
    registered = b"r" * 32
    wrong = tuple(bytes([index + 1]) * 32 for index in range(16))
    output = runtime.run_content_chain_unit(
        object(), unit, registered, wrong, _assets(), asset
    )
    assert len(pair_calls) == 1 and pair_calls[0]["seed"] == 7
    assert output.image is candidate and output.primary_null is primary_null
    assert output.measurement is measurement
    assert tuple(output.candidate_scores) == ("lf", "hf", "weighted_joint")
    assert tuple(output.primary_null_scores) == ("lf", "hf", "weighted_joint")
    assert len(score_calls) == 34
    assert [image for image, _ in score_calls[:17]] == [candidate] * 17
    assert [image for image, _ in score_calls[17:]] == [primary_null] * 17
    assert output.candidate_scores["weighted_joint"]["registered"] != min(
        output.candidate_scores["lf"]["registered"],
        output.candidate_scores["hf"]["registered"],
    )
